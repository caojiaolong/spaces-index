from types import SimpleNamespace

import pytest
import requests

from scripts import image_cache as images, extract_articles as extractor, mirror_articles as transport
from scripts.common import read_json, write_json
from scripts.mirror_content import MirrorError, digest
from scripts.mirror_store import prepare_images, verified_article, publish_mirrors, now_iso
from scripts.local_preview import export_preview
from test_extract_articles import page, response

PNG = b'\x89PNG\r\n\x1a\nfixture'
URL = 'https://spaces.ac.cn/usr/uploads/test.png'
MATPLOTLIB_SVG = b'''<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100" height="50">
 <metadata>
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
           xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#">
   <cc:Work><dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2026-06-26T01:02:42</dc:date><dc:format>image/svg+xml</dc:format>
    <dc:creator><cc:Agent><dc:title>Matplotlib v3.10.0, https://matplotlib.org/</dc:title></cc:Agent></dc:creator>
   </cc:Work>
  </rdf:RDF>
 </metadata>
 <defs><path id="line" d="M 0 0 L 100 50"/></defs><use xlink:href="#line"/>
</svg>'''
MATHJAX_DRAWIO_SVG = b'''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50">
 <style xmlns="http://www.w3.org/1999/xhtml"><![CDATA[
  .MathJax { display: inline; font-family: 'Times New Roman'; }
  @font-face { font-family: MathJax_Blank; src: url('about:blank') }
 ]]></style>
 <defs><clipPath id="clip"><rect width="100" height="50"/></clipPath></defs>
 <foreignObject width="100" height="50"><div xmlns="http://www.w3.org/1999/xhtml">
  <span style="font-style: italic"><nobr>x</nobr></span>
  <script type="math/tex; mode=display">x^2</script>
 </div></foreignObject>
 <path style="clip-path: url(#clip)" d="M 0 0 L 100 50"/>
</svg>'''


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / 'data/articles'
    monkeypatch.setattr(extractor, 'STORAGE_ROOT', tmp_path / 'data')
    monkeypatch.setattr(images, 'IMAGE_CACHE_DIR', tmp_path / '.cache/image-cache')
    monkeypatch.setattr('requests.Session.request', lambda *a, **k: pytest.fail('Unexpected network'))
    calls, responses, robots = [], [], []
    def factory(*args, host='spaces.ac.cn', **kwargs):
        def get(url, headers=None):
            calls.append((url, headers))
            result = responses.pop(0)
            if isinstance(result, BaseException): raise result
            return result
        return SimpleNamespace(get=get, check_robots=lambda: robots.append(host), session=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(transport, 'SerialFetcher', factory)
    monkeypatch.setattr(extractor, 'SerialFetcher', factory)
    def seed(post_id='12345', urls=(URL,)):
        source, meta = extractor.source_body(page(post_id, '<p>图片 $x$。</p>' + ''.join(f'<img src="{url}" alt="图">' for url in urls)), post_id)
        extractor.save_snapshot(root, post_id, str(source), meta)
        report = extractor.convert_snapshot(root, post_id)
        state = read_json(root / 'state.json', {})
        state[post_id] = {'status': 'verified', 'checked_at': now_iso(), 'markdown_sha256': report['validation']['markdown_sha256']}
        write_json(root / 'state.json', state)
        return root / post_id
    return root, seed, calls, responses, robots


def test_cache_is_per_article_deduplicated_and_resumable(library):
    root, seed, calls, responses, robots = library
    first = seed(urls=(URL, URL, 'https://other.example/x.png'))
    second = seed('12346')
    before = (first / 'article.md').read_bytes()
    responses.append(response(200, PNG, {'ETag': 'v1'}))
    result = images.sync_images(root, ['12345', '12346'], log=lambda m: None)
    assert result['downloaded'] == 1 and result['reused'] == 1 and len(calls) == 1
    for folder in (first, second):
        manifest = read_json(folder / 'images.json', {})['images']
        assert set(manifest) == {URL}
        assert images.cached_image_bytes(folder, manifest[URL]) == PNG
    assert (first / 'article.md').read_bytes() == before
    assert images.sync_images(root, ['12345', '12346'], log=lambda m: None) == {'cached': 2}
    assert robots == ['spaces.ac.cn']


def test_automatic_update_backfills_existing_body_and_offline_preserves_cache(library):
    root, seed, calls, responses, _ = library
    folder = seed()
    # Seed a complete metadata cache so no page request is needed.
    write_json(root / 'metadata.json', [{'id': 12345, 'source_category': '数学', 'source_tags': [], 'source_summary': None}])
    responses.append(response(200, PNG))
    assert extractor.main(['--ids', '12345']) == 0
    assert calls == [(URL, {})]
    original = (folder / 'images.json').read_bytes()
    assert extractor.main(['--ids', '12345', '--offline']) == 0
    assert (folder / 'images.json').read_bytes() == original
    assert len(calls) == 1


def test_conditional_refresh_and_updated_image_prunes_old_file(library):
    root, seed, calls, responses, _ = library
    folder = seed()
    responses.extend([response(200, PNG, {'ETag': 'v1'}), response(304), response(200, PNG + b'new', {'ETag': 'v2'})])
    images.sync_images(root, ['12345'], log=lambda m: None)
    first = next((folder / 'images').iterdir())
    result = images.sync_images(root, ['12345'], refresh_ids={'12345'}, log=lambda m: None)
    assert result['not_modified'] == 1 and calls[-1][1] == {'If-None-Match': 'v1'}
    images.sync_images(root, ['12345'], refresh_ids={'12345'}, log=lambda m: None)
    assert not first.exists() and len(list((folder / 'images').iterdir())) == 1


def test_broken_cache_is_refetched_without_conditional_headers(library):
    root, seed, calls, responses, _ = library
    folder = seed()
    responses.extend([response(200, PNG, {'ETag': 'v1'}), response(200, PNG)])
    images.sync_images(root, ['12345'], log=lambda m: None)
    next((folder / 'images').iterdir()).write_bytes(b'broken')
    images.sync_images(root, ['12345'], log=lambda m: None)
    assert calls[-1][1] == {}
    assert next((folder / 'images').iterdir()).read_bytes() == PNG


def test_reused_cache_does_not_skip_due_image_refresh(library):
    root, seed, calls, responses, _ = library
    seed()
    seed('12346')
    responses.append(response(200, PNG, {'ETag': 'v1'}))
    images.sync_images(root, ['12345', '12346'], log=lambda m: None)
    responses.append(response(304))
    result = images.sync_images(root, ['12345', '12346'], refresh_ids={'12346'}, log=lambda m: None)
    assert result['not_modified'] == 1 and len(calls) == 2
    assert calls[-1][1] == {'If-None-Match': 'v1'}


def test_removed_cache_does_not_restore_legacy_report_mapping(library):
    root, seed, _, _, _ = library
    folder = seed()
    report = read_json(folder / 'report.json', {})
    report['images'] = {URL: {'file': digest(PNG) + '.png', 'sha256': digest(PNG)}}
    write_json(folder / 'report.json', report)
    write_json(folder / 'images.json', {'version': 1, 'images': {URL: {'status': 'failed'}}})
    assert verified_article(root, '12345')['images'] == {}


def test_failure_isolated_and_retry_does_not_bypass_server_deferral(library):
    root, seed, calls, responses, _ = library
    folder = seed()
    responses.append(transport.DeferredRetry(86400))
    assert images.sync_images(root, ['12345'], log=lambda m: None)['failed'] == 1
    assert verified_article(root, '12345')['validation']['passed']
    assert images.sync_images(root, ['12345'], retry_failed=True, log=lambda m: None)['deferred'] == 1
    assert len(calls) == 1
    assert not (folder / 'images').exists()


def test_transient_refresh_failure_keeps_previous_valid_file(library):
    root, seed, _, responses, _ = library
    folder = seed()
    responses.extend([response(200, PNG), requests.Timeout('timeout')])
    images.sync_images(root, ['12345'], log=lambda m: None)
    result = images.sync_images(root, ['12345'], refresh_ids={'12345'}, log=lambda m: None)
    assert result['failed'] == 1
    entry = read_json(folder / 'images.json', {})['images'][URL]
    assert entry['status'] == 'cached' and entry['error']
    assert images.cached_image_bytes(folder, entry) == PNG
    responses.append(response(200, PNG))
    images.sync_images(root, ['12345'], retry_failed=True, log=lambda m: None)
    assert 'error' not in read_json(folder / 'images.json', {})['images'][URL]


def test_external_redirect_never_requested(library):
    root, seed, calls, responses, _ = library
    seed()
    responses.append(response(302, headers={'Location': 'https://third.example/x.png'}))
    assert images.sync_images(root, ['12345'], log=lambda m: None)['failed'] == 1
    assert [u for u, _ in calls] == [URL]


def test_same_host_redirect_and_subdomain_each_check_robots(library):
    root, seed, calls, responses, robots = library
    seed(urls=(URL, 'http://album.spaces.ac.cn/a.jpg'))
    responses.extend([response(302, headers={'Location': '/moved.png'}), response(200, PNG), response(200, b'\xff\xd8\xffdata')])
    images.sync_images(root, ['12345'], log=lambda m: None)
    assert [u for u, _ in calls] == [URL, 'https://spaces.ac.cn/moved.png', 'https://album.spaces.ac.cn/a.jpg']
    assert robots == ['spaces.ac.cn', 'album.spaces.ac.cn']


def test_bounded_trial_and_ordinary_failure_can_resume(library):
    root, seed, calls, responses, _ = library
    seed(urls=(URL, URL + '?second'))
    responses.append(response(404))
    result = images.sync_images(root, ['12345'], max_requests=1, log=lambda m: None)
    assert result['failed'] == 1 and result['pending'] == 1
    responses.extend([response(200, PNG), response(200, PNG)])
    result = images.sync_images(root, ['12345'], retry_failed=True, log=lambda m: None)
    assert result['downloaded'] == 2 and len(calls) == 3


def test_three_consecutive_failures_pause_only_that_host(library):
    root, seed, calls, responses, _ = library
    for i in range(4): seed(str(12345 + i), urls=(URL + str(i),))
    seed('12349', urls=('https://album.spaces.ac.cn/healthy.png',))
    responses.extend([requests.Timeout()] * 3 + [response(200, PNG)])
    result = images.sync_images(root, [str(12345+i) for i in range(4)] + ['12349'], log=lambda m: None)
    assert result['hosts_paused'] == 1 and result['host_deferred'] == 1
    assert result['failed'] == 3 and result['downloaded'] == 1 and len(calls) == 4
    assert 'stopped_early' not in result


def test_robots_tls_failure_is_checked_once_and_other_hosts_continue(library, monkeypatch):
    root, seed, calls, responses, robots = library
    album = 'https://album.spaces.ac.cn/pictures/photo.png'
    seed(urls=(album, album+'?second', URL))
    seed('12346', urls=(album+'?third',))
    original = transport.SerialFetcher
    closed = []
    def factory(*args, host='spaces.ac.cn', **kwargs):
        fetcher = original(*args, host=host, **kwargs)
        fetcher.session.close = lambda: closed.append(host)
        if host == 'album.spaces.ac.cn':
            def check():
                robots.append(host)
                raise requests.exceptions.SSLError('UNEXPECTED_EOF_WHILE_READING /robots.txt')
            fetcher.check_robots = check
        return fetcher
    monkeypatch.setattr(transport, 'SerialFetcher', factory)
    responses.append(response(200, PNG))
    result = images.sync_images(root, ['12345', '12346'], log=lambda m: None)
    assert result['failed'] == 1 and result['hosts_paused'] == 1 and result['host_deferred'] == 2
    assert result['downloaded'] == 1 and calls == [(URL, {})]
    assert robots.count('album.spaces.ac.cn') == 1
    assert sorted(closed) == ['album.spaces.ac.cn', 'spaces.ac.cn']
    # Persist the host cooldown even for images that have never been attempted.
    result = images.sync_images(root, ['12346'], log=lambda m: None)
    assert result['deferred'] == 1 and robots.count('album.spaces.ac.cn') == 1
    # Ordinary errors can be retried explicitly; a recovered host stays usable.
    monkeypatch.setattr(transport, 'SerialFetcher', original)
    responses.append(response(200, PNG))
    assert images.sync_images(root, ['12346'], retry_failed=True, log=lambda m: None)['downloaded'] == 1
    seed('12347', urls=(album+'?fourth',))
    responses.append(response(200, PNG))
    assert images.sync_images(root, ['12347'], log=lambda m: None)['downloaded'] == 1


def test_server_deferral_covers_new_urls_and_redirects_across_runs(library):
    root, seed, calls, responses, _ = library
    album = 'https://album.spaces.ac.cn/picture.png'
    seed(urls=(album, URL))
    responses.extend([transport.DeferredRetry(86400), response(200, PNG)])
    result = images.sync_images(root, ['12345'], log=lambda m: None)
    assert result['failed'] == 1 and result['downloaded'] == 1
    seed('12346', urls=(album+'?new', URL+'?redirect', URL+'?healthy'))
    responses.extend([response(302, headers={'Location': album+'?redirected'}), response(200, PNG)])
    result = images.sync_images(root, ['12346'], retry_failed=True, log=lambda m: None)
    assert result['host_deferred'] == 2 and result['downloaded'] == 1
    assert [url for url, _ in calls] == [album, URL, URL+'?redirect', URL+'?healthy']
    redirected = read_json(root/'12346/images.json', {})['images'][URL+'?redirect']
    assert redirected['server_deferred']


def test_redirect_failure_pauses_target_not_healthy_source(library, monkeypatch):
    root, seed, calls, responses, robots = library
    seed(urls=(URL+'?redirect', URL+'?healthy'))
    original = transport.SerialFetcher
    def factory(*args, host='spaces.ac.cn', **kwargs):
        fetcher = original(*args, host=host, **kwargs)
        if host == 'album.spaces.ac.cn':
            def check():
                robots.append(host)
                raise requests.exceptions.SSLError('TLS failure')
            fetcher.check_robots = check
        return fetcher
    monkeypatch.setattr(transport, 'SerialFetcher', factory)
    responses.extend([response(302, headers={'Location': 'https://album.spaces.ac.cn/picture.png'}), response(200, PNG)])
    result = images.sync_images(root, ['12345'], log=lambda m: None)
    assert result['failed'] == 1 and result['downloaded'] == 1
    state = read_json(next(images.IMAGE_CACHE_DIR.glob('hosts-*.json')), {})
    assert set(state) == {'album.spaces.ac.cn'}


def test_cli_partial_build_does_not_report_all_images_completed(library, tmp_path, monkeypatch, capsys):
    from scripts import cache_images as cli, build_site
    root, seed, calls, responses, _ = library
    seed(urls=(URL, URL+'?2', URL+'?3', URL+'?4'))
    responses.extend([requests.exceptions.SSLError('UNEXPECTED_EOF_WHILE_READING')] * 3)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'ARTICLES_DIR', root)
    monkeypatch.setattr(cli, 'load_config', lambda: {})
    builds = []
    monkeypatch.setattr(build_site, 'build_site', lambda *a, **kw: builds.append((a,kw)))
    assert cli.main(['--all', '--retry-failed', '--build']) == 1
    assert len(calls) == 3 and len(builds) == 2
    summary = read_json(tmp_path/'.cache/image-cache/summary.json', {})
    assert summary['failed'] == 3 and summary['hosts_paused'] == 1 and summary['host_deferred'] == 1
    assert verified_article(root,'12345')['validation']['passed']
    output = capsys.readouterr().out
    assert 'continuing other source hosts' in output
    assert output.rstrip().endswith('image caching is still incomplete.')


def test_cli_host_cooldown_reports_incomplete_without_new_requests(library, tmp_path, monkeypatch, capsys):
    from scripts import cache_images as cli
    root, seed, calls, responses, _ = library
    seed()
    responses.append(transport.DeferredRetry(86400))
    images.sync_images(root, ['12345'], log=lambda m: None)
    seed('12346', urls=(URL+'?unvisited',))
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'ARTICLES_DIR', root)
    monkeypatch.setattr(cli, 'load_config', lambda: {})
    assert cli.main(['--ids', '12346', '--retry-failed']) == 1
    assert len(calls) == 1
    summary = read_json(tmp_path/'.cache/image-cache/summary.json', {})
    assert summary['deferred'] == 1 and not summary.get('failed')
    assert 'Image caching is incomplete' in capsys.readouterr().out


def test_public_and_preview_export_same_valid_files(library, tmp_path):
    root, seed, _, responses, _ = library
    folder = seed()
    responses.append(response(200, PNG))
    images.sync_images(root, ['12345'], log=lambda m: None)
    public, preview = tmp_path / 'public', tmp_path / 'preview'
    publish_mirrors(public, root, allowed_ids={'12345'})
    export_preview(preview, root, {'12345'})
    mapping = read_json(public / 'mirror/12345/article.json', {})['images']
    assert mapping == read_json(preview / 'mirror/12345/article.json', {})['images']
    assert (public / mapping[URL]).read_bytes() == (preview / mapping[URL]).read_bytes() == PNG
    assert (public / 'mirror/12345/article.md').read_bytes() == (folder / 'article.md').read_bytes()
    next((folder / 'images').iterdir()).write_bytes(b'corrupt')
    with pytest.raises(MirrorError): publish_mirrors(tmp_path / 'bad', root, allowed_ids={'12345'})
    assert not (tmp_path / 'bad/mirror/12345/article.md').exists()


def test_paths_and_svg_content_are_checked(library):
    root, seed, _, _, _ = library
    folder = seed()
    with pytest.raises(MirrorError): images.cached_image_bytes(folder, {'file': '../secret', 'sha256': 'x'})
    for url in ('https://spaces.ac.cn.evil.test/a.png','https://spaces.ac.cn@evil.test/x','https://spaces.ac.cn:99/x','file:///x'):
        assert images.hosted_image_url(url) is None
    assert images.image_extension(b'<svg xmlns="http://www.w3.org/2000/svg"><path fill="url(#id)"/></svg>') == '.svg'
    label = b'<svg xmlns="http://www.w3.org/2000/svg"><foreignObject><div xmlns="http://www.w3.org/1999/xhtml"><b>Label</b></div></foreignObject><a href="https://example.com/help"><text>Help</text></a></svg>'
    assert images.image_extension(label) == '.svg'
    for raw in (b'<html>gate</html>',b'<svg><script>alert(1)</script></svg>',b'<svg><image href="https://x/x.png"/></svg>',b'<svg><style>@import "x";</style></svg>',
                b'<svg><foreignObject><iframe xmlns="http://www.w3.org/1999/xhtml"/></foreignObject></svg>',
                b'<svg><a href="javascript:alert(1)">x</a></svg>'):
        with pytest.raises(MirrorError): images.image_extension(raw)


@pytest.mark.parametrize('raw', [MATPLOTLIB_SVG, MATHJAX_DRAWIO_SVG], ids=['matplotlib', 'mathjax-drawio'])
def test_svg_exporter_formats_retry_and_export_unchanged(library, tmp_path, raw):
    root, seed, calls, responses, _ = library
    url = URL.removesuffix('.png') + '.svg'
    folder = seed(urls=(url,))
    write_json(folder / 'images.json', {'version': 1, 'images': {url: {
        'status': 'failed', 'checked_at': now_iso(), 'error': 'Unsupported embedded SVG content',
    }}})
    responses.append(response(200, raw))
    result = images.sync_images(root, ['12345'], retry_failed=True, log=lambda m: None)
    assert result == {'requests': 1, 'downloaded': 1}
    entry = read_json(folder / 'images.json', {})['images'][url]
    assert entry['status'] == 'cached' and 'error' not in entry
    assert images.cached_image_bytes(folder, entry) == raw
    assert images.sync_images(root, ['12345'], log=lambda m: None) == {'cached': 1}
    assert len(calls) == 1
    for output, export in ((tmp_path / 'public', publish_mirrors), (tmp_path / 'preview', export_preview)):
        export(output, root, allowed_ids={'12345'})
        mapping = read_json(output / 'mirror/12345/article.json', {})['images']
        assert (output / mapping[url]).read_bytes() == raw


@pytest.mark.parametrize('css', [
    'path { fill: red; stroke: url(#line) }',
    '@font-face { font-family: MathJax_Blank; src: url("about:blank") }',
    '@media screen { path { fill: url("#gradient") } }',
    r'path { fill: u\72l("#gradient") }',
    '/* @import "https://example.com/comment"; */ path { fill: red }',
])
def test_xhtml_styles_preserve_static_css(css):
    raw = ('<svg xmlns="http://www.w3.org/2000/svg"><style xmlns="http://www.w3.org/1999/xhtml"><![CDATA['
           + css + ']]></style></svg>').encode()
    assert images.image_extension(raw) == '.svg'


@pytest.mark.parametrize('css', [
    '@import "https://example.com/x.css";',
    r'@\69mport "https://example.com/x.css";',
    'path { fill: url(https://example.com/x.svg) }',
    r'path { fill: u\72l("https://example.com/x.svg") }',
    '@font-face { font-family: remote; src: url("https://example.com/f.woff") }',
    '@media screen { path { fill: url("https://example.com/x.svg") } }',
    'div { background: image-set("https://example.com/x.png" 1x) }',
    'div { background: url("data:text/html,payload") }',
    'div { width: expression(alert(1)) }',
    'div { behavior: url(#binding) }',
])
def test_xhtml_styles_still_reject_external_and_active_css(css):
    raw = ('<svg xmlns="http://www.w3.org/2000/svg"><style xmlns="http://www.w3.org/1999/xhtml"><![CDATA['
           + css + ']]></style></svg>').encode()
    with pytest.raises(MirrorError): images.image_extension(raw)


@pytest.mark.parametrize('content', [
    '<script xmlns="http://www.w3.org/1999/xhtml">alert(1)</script>',
    '<script xmlns="http://www.w3.org/1999/xhtml" type="text/javascript">alert(1)</script>',
    '<script xmlns="http://www.w3.org/1999/xhtml" type="math/tex" src="https://example.com/a.js"/>',
    '<script xmlns="http://www.w3.org/1999/xhtml" type="math/tex" onclick="alert(1)"/>',
    '<script type="math/tex">SVG scripts are not HTML data blocks</script>',
    '<script xmlns="http://www.w3.org/1999/xhtml" type="math/tex"><script>alert(1)</script></script>',
    '<rect fill="u&#92;72l(https://example.com/x.svg)"/>',
])
def test_mathjax_support_still_rejects_active_content(content):
    with pytest.raises(MirrorError):
        images.image_extension(('<svg xmlns="http://www.w3.org/2000/svg">'+content+'</svg>').encode())


@pytest.mark.parametrize('content', [
    b'<metadata><script>alert(1)</script></metadata>',
    b'<metadata><dc:title xmlns:dc="http://purl.org/dc/elements/1.1/" onclick="alert(1)"/></metadata>',
    b'<metadata><image href="https://example.com/remote.png"/></metadata>',
    b'<metadata><style>@import "https://example.com/remote.css";</style></metadata>',
    b'<metadata><custom:data xmlns:custom="urn:unknown"/></metadata>',
    b'<dc:title xmlns:dc="http://purl.org/dc/elements/1.1/">Outside metadata</dc:title>',
])
def test_metadata_support_does_not_bypass_svg_checks(content):
    with pytest.raises(MirrorError):
        images.image_extension(b'<svg xmlns="http://www.w3.org/2000/svg">' + content + b'</svg>')


def test_removal_clears_only_generated_images(library):
    root, seed, _, responses, _ = library
    folder = seed()
    responses.append(response(200, PNG))
    images.sync_images(root, ['12345'], log=lambda m: None)
    (folder / 'images/notes.txt').write_text('keep')
    extractor.remove_images(folder)
    assert not (folder / 'images.json').exists()
    assert [f.name for f in (folder / 'images').iterdir()] == ['notes.txt']


def test_streamed_image_limit_closes_response():
    closed, waits = [], []
    response_stream = SimpleNamespace(iter_content=lambda size: iter([b'abcd', b'efgh']), close=lambda: closed.append(True))
    def get(*args, **kwargs):
        assert kwargs['stream'] and not kwargs['allow_redirects']
        return response_stream
    fetcher = transport.SerialFetcher(max_bytes=5, session=SimpleNamespace(headers={}, get=get), sleep=waits.append)
    with pytest.raises(MirrorError, match='size limit'):
        fetcher.get(URL)
    assert closed == [True] and waits == [3]


def test_robots_delay_and_disallow_override_three_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(transport, 'CACHE', tmp_path)
    write_json(tmp_path / 'robots.json', {'checked_at': now_iso(), 'text': 'User-agent: *\nCrawl-delay: 9\nDisallow: /blocked/\n'})
    waits, calls = [], []
    fetcher = transport.SerialFetcher(session=SimpleNamespace(headers={}, get=lambda url, **kw: calls.append(url) or response(200, PNG)), sleep=waits.append)
    fetcher.check_robots()
    fetcher.get(URL)
    assert waits == [9]
    with pytest.raises(MirrorError, match='disallows'):
        fetcher.get('https://spaces.ac.cn/blocked/image.png')
    assert calls == [URL]
