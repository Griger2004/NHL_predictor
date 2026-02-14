import asyncio
import aiohttp


class ApiClient:
    def __init__(
        self,
        base_url,
        timeout,
        max_concurrent_requests,
        retries,
        retry_statuses=None,
    ):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._retries = retries
        self._retry_statuses = retry_statuses or {429, 500, 502, 503, 504}
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._session = None

    async def __aenter__(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def get_json(self, path_or_url):
        if self._session is None:
            raise RuntimeError("ApiClient must be used with 'async with'.")

        url = self._build_url(path_or_url)

        async with self._semaphore:
            for attempt in range(self._retries):
                try:
                    async with self._session.get(url) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        if resp.status in self._retry_statuses:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return None
                except Exception:
                    await asyncio.sleep(2 ** attempt)

        return None

    def _build_url(self, path_or_url):
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        if not path_or_url.startswith("/"):
            path_or_url = f"/{path_or_url}"
        return f"{self._base_url}{path_or_url}"
