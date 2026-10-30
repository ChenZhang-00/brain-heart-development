"""Build the Apple Silicon app from the shared editor source."""

from importlib.metadata import distributions
from pathlib import Path

here = Path(SPECPATH)
editor_dir = here.parent
datas = [(str(editor_dir.parents[2] / "LICENSE"), ".")]

# Keep dependency licenses with the application.
for package in distributions():
    for file in package.files or []:
        if any(word in file.name.lower() for word in ("license", "copying", "notice")):
            source = package.locate_file(file)
            if source.is_file():
                destination = Path("licenses") / package.metadata["Name"] / file.parent
                datas.append((str(source), str(destination)))

a = Analysis(
    [str(editor_dir / "launch_editor.py")],
    pathex=[str(editor_dir)],
    datas=datas,
    hooksconfig={"matplotlib": {"backends": ["TkAgg"]}},
    excludes=["IPython", "pytest"],
)
# Matplotlib's example datasets are not needed by the editor.
a.datas = [entry for entry in a.datas if "sample_data" not in Path(entry[0]).parts]
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RpeakEditor",
    console=False,
    target_arch="arm64",
)
collection = COLLECT(exe, a.binaries, a.datas, name="RpeakEditor")
app = BUNDLE(
    collection,
    name="RpeakEditor.app",
    icon=str(here / "RpeakEditor.icns"),
    bundle_identifier="org.dbh.rpeakeditor",
    version="2026.10.25",
    info_plist={"LSMinimumSystemVersion": "14.0", "NSHighResolutionCapable": True},
)
