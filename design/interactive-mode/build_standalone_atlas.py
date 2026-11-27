"""Build the downloadable atlas from the same modules and measured data as the local app.

Run with Python 3.9+ from any directory. Opens in a modern browser with no server.
"""

import base64
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATLAS = HERE.parents[1] / "src/sctrial/interactive/atlas"
WORKSHEET = HERE / "sctrial-analysis-worksheet.html"
OUTPUT = HERE / "sctrial-interactive-preview.html"
MIME = {
    ".bin": "application/octet-stream",
    ".gmt": "text/plain",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".ttf": "font/ttf",
    ".wasm": "application/wasm",
}
LOCAL_LITERAL = re.compile(r"""(?P<quote>["'])(?P<path>\.{1,2}/[^\s"'\\]+)(?P=quote)""")

# Evaluate imports with native module semantics in both the document and worker.
MODULE_LOADER = """
const urls = new Map();
async function makeURL(name) {
  if (urls.has(name)) return urls.get(name);
  const generating = (async () => {
    const item = archive[name];
    const compressed = Uint8Array.from(atob(item.bytes), char => char.charCodeAt(0));
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate'));
    const data = await new Response(stream).arrayBuffer();
    let contents = data;
    if (item.dependencies.length) {
      let source = new TextDecoder().decode(data);
      for (const [marker, dependency] of item.dependencies) {
        source = source.replaceAll(marker, await makeURL(dependency));
      }
      contents = source;
    }
    const blob = new Blob([contents], {type: item.type});
    // Keep the filename for the example's extension-based decoder.
    return URL.createObjectURL(blob) + '#' + name;
  })();
  urls.set(name, generating);
  return generating;
}
"""

BOOT = (
    """
<script type="module">
const archive = JSON.parse(document.getElementById('atlasArchive').textContent);
"""
    + MODULE_LOADER
    + """
try {
  const worksheet = await makeURL('worksheet.html');
  document.querySelector('[data-worksheet-open]').href = worksheet;
  document.querySelector('[data-worksheet-download]').href = worksheet;
  await import(await makeURL('app.js'));
} catch (error) {
  document.getElementById('busy').textContent = 'The offline atlas could not open. ' + error.message;
  console.error(error);
}
</script>
"""
)


def pack(assets, name, data, dependencies=()):
    assets[name] = {
        "type": MIME[Path(name).suffix],
        "bytes": base64.b64encode(zlib.compress(data, 9)).decode("ascii"),
        "dependencies": dependencies,
    }


def build():
    assets = {}
    visiting = set()

    def collect(path):
        name = path.relative_to(ATLAS).as_posix()
        if name in assets:
            return name
        if name in visiting:
            raise ValueError(f"Cyclic module reference at {name}")
        visiting.add(name)
        dependencies = []
        data = path.read_bytes()
        # Upstream Emscripten's prebuilt module has its own embedded WASM.
        if path.suffix == ".js" and name != "vendor/hdf5_util.js":

            def substitute(match):
                target = (path.parent / match["path"]).resolve(strict=True)
                dep = collect(target)
                marker = f"__SCTRIAL_BUNDLE_{len(dependencies)}__"
                dependencies.append((marker, dep))
                return match["quote"] + marker + match["quote"]

            source = LOCAL_LITERAL.sub(substitute, data.decode("utf-8"))
            if any(dep == "import-worker.js" for _, dep in dependencies):
                # Chromium disallows module blob workers from file://. The classic
                # bootstrap imports the unchanged application modules inside the worker.
                source = (
                    "class Worker extends globalThis.Worker {constructor(url){super(url)}}\n"
                    + source
                )
            data = source.encode("utf-8")
        pack(assets, name, data, dependencies)
        visiting.remove(name)
        return name

    collect(ATLAS / "import-worker.js")
    worker = "const archive=" + json.dumps(assets, separators=(",", ":"), sort_keys=True) + ";\n"
    worker += (
        MODULE_LOADER
        + """
// Queue the first File message during asynchronous module initialization.
const pending = [];
self.onmessage = event => pending.push(event);
(async () => {
  try {
    await import(await makeURL('import-worker.js'));
    for (const event of pending.splice(0)) self.onmessage(event);
  } catch (error) {
    self.onmessage = ({data}) => self.postMessage({id: data.id, error: error.message});
    for (const event of pending.splice(0)) self.onmessage(event);
  }
})();
"""
    )
    assets.clear()
    pack(assets, "import-worker.js", worker.encode("utf-8"))
    collect(ATLAS / "app.js")
    pack(assets, "worksheet.html", WORKSHEET.read_bytes())
    template = (ATLAS / "index.html").read_text(encoding="utf-8")

    def inline_styles(match):
        sheet = ATLAS / match[1]

        def inline_url(url):
            resource = (sheet.parent / url[1]).resolve(strict=True)
            encoded = base64.b64encode(resource.read_bytes()).decode("ascii")
            return f"url('data:{MIME[resource.suffix]};base64,{encoded}')"

        css = re.sub(r"""url\(\s*['"]?(\./[^)'"]+)['"]?\s*\)""", inline_url, sheet.read_text())
        return "<style>\n" + css + "\n</style>"

    template = re.sub(
        r'<link\s+rel="stylesheet"\s+href="(\./[^"]+)"\s*/?>', inline_styles, template
    )
    template = template.replace(
        'href="./workspace.html"',
        'href="#worksheet" data-worksheet-open target="_blank" rel="noopener"',
    )
    template = template.replace(
        '<div id="studyLinks"',
        '<div class="note-links"><a href="#worksheet" data-worksheet-download download="sctrial-analysis-worksheet.html">Download analysis worksheet</a></div>\n  <div id="studyLinks"',
    )
    if "data-worksheet-open" not in template or "data-worksheet-download" not in template:
        raise ValueError("Cannot locate the worksheet navigation in the atlas template")
    notices = []
    for path in sorted([*(ATLAS / "vendor").glob("*.txt"), *(ATLAS / "fonts").glob("*.txt")]):
        notices.append(f"{path.name}\n\n{path.read_text(encoding='utf-8')}")
    license_text = "\n".join(line.rstrip() for line in "\n\n".join(notices).splitlines())
    license_text = license_text.replace("</script", "<\\/script")
    payload = json.dumps(assets, separators=(",", ":"), sort_keys=True)
    scripts = (
        f'<script type="text/plain" id="thirdPartyNotices">{license_text}</script>\n'
        f'<script type="application/json" id="atlasArchive">{payload}</script>\n' + BOOT
    )
    template, count = re.subn(
        r'<script\s+type="module"\s+src="\./app\.js"\s*>\s*</script>', lambda _: scripts, template
    )
    if count != 1:
        raise ValueError("Cannot locate the atlas entry module")
    return template.encode("utf-8")


if __name__ == "__main__":
    OUTPUT.write_bytes(build())
    print(f"Wrote {OUTPUT.name} ({OUTPUT.stat().st_size:,} bytes)")
