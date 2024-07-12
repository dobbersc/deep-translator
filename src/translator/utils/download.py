import http.client
import urllib.request
from pathlib import Path
from typing import Any

from tqdm import tqdm


class DownloadProgressBar(tqdm):  # type: ignore[type-arg]
    """TQDM progress bar that provides an update hook based on a download's progress."""

    def __init__(
        self,
        *,
        unit: str = "B",
        unit_scale: bool | float = True,
        unit_divisor: float = 1024,
        miniters: int = 1,
        **kwargs: Any,
    ) -> None:
        if "iterable" in kwargs:
            msg: str = f"{type(self).__name__} does not support iterables directly. Use the `update_to` hook instead."
            raise ValueError(msg)

        super().__init__(
            iterable=None,
            unit=unit,
            unit_scale=unit_scale,
            unit_divisor=unit_divisor,
            miniters=miniters,
            **kwargs,
        )

    def update_to(self, blocks_transferred: int = 1, block_size: int = 1, total_size: int | None = None) -> bool | None:
        """Updates the progress bar's progress in relation to the download state.

        Args:
            blocks_transferred: The number of blocks transferred.
            block_size: The size of each block.
            total_size: The total size of the file.

        Returns:
            The result of the tqdm update (see `tqdm.update`).
        """
        if total_size is not None:
            self.total = total_size
        return self.update(blocks_transferred * block_size - self.n)


def download_from_url(url: str, destination: Path | None = None) -> tuple[Path, http.client.HTTPMessage]:
    with DownloadProgressBar() as progress_bar:
        result_path, headers = urllib.request.urlretrieve(  # noqa: S310
            url,
            destination,
            reporthook=progress_bar.update_to,
        )
    return Path(result_path), headers
