import pytest
import requests

from scripts.common import REQUEST_HEADERS, fetch_html_with_retries


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "<html>ok</html>"):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if 400 <= self.status_code:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class FakeSession:
    def __init__(self, *results: object):
        self.results = list(results)
        self.headers: dict[str, str] = {}
        self.calls = 0
        self.last_timeout: float | None = None

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls += 1
        self.last_timeout = timeout
        if not self.results:
            raise AssertionError(f"Unexpected extra request to {url}")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def fetch_with_fake_session(session: FakeSession, url: str = "https://spaces.ac.cn/content.html") -> str:
    return fetch_html_with_retries(
        session,  # type: ignore[arg-type]
        url,
        page_name="test page",
        retry_delays=(0.0,),
    )


def test_fetch_retries_connection_error_then_succeeds():
    session = FakeSession(
        requests.ConnectionError("remote closed"),
        FakeResponse(text="<html>archive</html>"),
    )

    assert fetch_with_fake_session(session) == "<html>archive</html>"
    assert session.calls == 2
    assert session.last_timeout == 30.0
    assert session.headers["User-Agent"] == REQUEST_HEADERS["User-Agent"]


def test_fetch_fails_after_all_retryable_exceptions():
    url = "https://spaces.ac.cn/content.html"
    session = FakeSession(
        requests.ConnectionError("remote closed"),
        requests.ConnectionError("remote closed"),
        requests.ConnectionError("remote closed"),
        requests.ConnectionError("remote closed"),
        requests.ConnectionError("remote closed"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        fetch_with_fake_session(session, url)

    message = str(exc_info.value)
    assert url in message
    assert "after 5 attempts" in message
    assert "ConnectionError" in message
    assert session.calls == 5


def test_fetch_retries_gate_page_then_succeeds():
    gate_page = "<script>window.location.href = '/content.html'</script>"
    session = FakeSession(
        FakeResponse(text=gate_page),
        FakeResponse(text="<html>archive</html>"),
    )

    assert fetch_with_fake_session(session) == "<html>archive</html>"
    assert session.calls == 2


def test_fetch_does_not_retry_404():
    session = FakeSession(FakeResponse(status_code=404, text="<html>not found</html>"))

    with pytest.raises(RuntimeError) as exc_info:
        fetch_with_fake_session(session)

    assert "attempt 1/5" in str(exc_info.value)
    assert "HTTP 404" in str(exc_info.value)
    assert session.calls == 1


def test_fetch_retries_503_then_succeeds():
    session = FakeSession(
        FakeResponse(status_code=503, text="<html>temporary</html>"),
        FakeResponse(text="<html>archive</html>"),
    )

    assert fetch_with_fake_session(session) == "<html>archive</html>"
    assert session.calls == 2
