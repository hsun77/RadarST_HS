"""Optional allowlisted Python IO and explicit native HDF5 auditing."""
import json
import os
from pathlib import Path
import platform
import sys


def install(output, inputs):
    platform.uname()
    output = Path(output).resolve()
    inputs = {Path(p).resolve() for p in inputs if p}
    runtime = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), Path(os.environ.get("WINDIR", "/usr")).resolve()]
    code = Path(__file__).resolve().parent
    log = (output / "file_access.jsonl").open("w", encoding="utf-8")
    seen = set()
    def check(path, write=False, event="open", detail=None):
        if isinstance(path, int):
            return
        text = os.fsdecode(path)
        if text.lower() in {"nul", "/dev/null"}:
            return
        p = Path(text).absolute().resolve()
        own = p.is_relative_to(output)
        source = p.is_relative_to(code) and p.suffix.lower() in {".py", ".pyc", ".pyd", ".so"}
        ok = own if write else own or p in inputs or source or any(p.is_relative_to(q) for q in runtime)
        key = (str(p), write, event, detail)
        if key not in seen:
            seen.add(key)
            log.write(json.dumps(dict(path=str(p),write=write,event=event,detail=detail,allowed=ok))+"\n")
            log.flush()
        if not ok:
            raise RuntimeError("Undeclared file access blocked: " + str(p))
    def hook(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            mode, flags = args[1:3]
            write = isinstance(mode, str) and any(c in mode for c in "wax+") or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            check(args[0], write, event)
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename"} and isinstance(args[0], (str, bytes)):
            check(args[0], True, event)
            if event == "os.rename":
                check(args[1], True, event)
        elif event in {"subprocess.Popen", "os.system"}:
            raise RuntimeError("Child process blocked during audited inference")
    sys.addaudithook(hook)
    import h5py
    old_init, old_get = h5py.File.__init__, h5py.Dataset.__getitem__
    def init(self, name, mode="r", *a, **kw):
        if isinstance(name,(str,bytes,os.PathLike)):
            check(name, mode != "r", "HDF5.File")
        return old_init(self,name,mode,*a,**kw)
    def get(self, key, *a, **kw):
        if self.name.startswith("/obs/") and self.name.split("/")[2] not in {"_index","index","barcode","spot_id"}:
            raise RuntimeError("Pathology observation metadata forbidden during inference")
        check(self.file.filename,False,"HDF5.Dataset",self.name)
        return old_get(self,key,*a,**kw)
    h5py.File.__init__, h5py.Dataset.__getitem__ = init, get
    return log
