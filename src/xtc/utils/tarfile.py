#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import tarfile
from typing import Any
from typing_extensions import override


class TarFile(tarfile.TarFile):
    """Default TarFile class interface is not compatible anymore with
    python 3.10 due to new filter argument.
    This is a wrapper class to make it work for python 3.10-3.14+.

    Use it as:

        from xtc.utils.tar import TarFile
        with TarFile.open(fname) as tf:
            tf.extractall(tmpdir, filter="data")
    """

    @override
    def extractall(  # type: ignore
        self, *args: Any, **kwargs: Any
    ):
        if kwargs.get("filter") is None:
            kwargs["filter"] = "data"
        try:
            return super().extractall(*args, **kwargs)
        except TypeError:
            del kwargs["filter"]
            super().extractall(*args, **kwargs)

    @override
    def extract(  # type: ignore
        self, *args: Any, **kwargs: Any
    ):
        if kwargs.get("filter") is None:
            kwargs["filter"] = "data"
        try:
            return super().extract(*args, **kwargs)
        except TypeError:
            del kwargs["filter"]
            super().extract(*args, **kwargs)
