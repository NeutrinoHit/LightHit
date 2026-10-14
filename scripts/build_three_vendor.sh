#!/bin/sh
# Rebuild src/lighthit/assets/vendor/three.min.js, the pinned three.js bundle
# that the default event viewer inlines.  It holds the three.js core classes
# the viewer uses plus OrbitControls, EffectComposer, RenderPass, ShaderPass,
# Pass and FullScreenQuad, as one minified classic script defining ``THREE``.
# Needs Node.js and network access to the npm registry; run from anywhere.
set -eu
THREE_VERSION=0.160.1
ESBUILD_VERSION=0.24.0
root=$(cd "$(dirname "$0")/.." && pwd)
vendor="$root/src/lighthit/assets/vendor"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
npm init -y >/dev/null
npm install --silent --no-audit --no-fund --save-exact "three@$THREE_VERSION" "esbuild@$ESBUILD_VERSION"
cat > entry.js <<'EOF'
export {REVISION, WebGLRenderer, WebGLRenderTarget, Scene, PerspectiveCamera, Mesh, BufferGeometry,
  BufferAttribute, InstancedBufferGeometry, InstancedInterleavedBuffer, InterleavedBufferAttribute,
  RawShaderMaterial, ShaderMaterial, GLSL3, DoubleSide, NoBlending, CustomBlending, AddEquation,
  OneFactor, OneMinusSrcAlphaFactor, DynamicDrawUsage, HalfFloatType, UnsignedByteType,
  LinearSRGBColorSpace, NoToneMapping, Vector2, Vector3, Matrix4, MOUSE, TOUCH} from "three";
export {OrbitControls} from "three/addons/controls/OrbitControls.js";
export {EffectComposer} from "three/addons/postprocessing/EffectComposer.js";
export {RenderPass} from "three/addons/postprocessing/RenderPass.js";
export {ShaderPass} from "three/addons/postprocessing/ShaderPass.js";
export {Pass, FullScreenQuad} from "three/addons/postprocessing/Pass.js";
EOF
license=node_modules/three/LICENSE
{
  printf '/*! three.js r%s (npm three@%s) with OrbitControls, EffectComposer, RenderPass,\n' \
    "${THREE_VERSION#0.}" "$THREE_VERSION" | sed 's/r\([0-9]*\)\.[0-9]*/r\1/'
  printf ' * ShaderPass and Pass, bundled by esbuild %s with\n' "$ESBUILD_VERSION"
  printf ' * scripts/build_three_vendor.sh.  https://threejs.org\n *\n'
  sed 's/^/ * /; s/ *$//' "$license"
  printf ' */\n'
} > banner.txt
npx esbuild entry.js --bundle --minify --format=iife --global-name=THREE --target=es2020 \
  --legal-comments=none --log-level=warning --outfile=bundle.js
mkdir -p "$vendor"
cat banner.txt bundle.js > "$vendor/three.min.js"
cp "$license" "$vendor/three.LICENSE"
if grep -qi '</script\|<!--\|<script' "$vendor/three.min.js"; then
  echo "bundle contains an HTML script delimiter" >&2
  exit 1
fi
wc -c "$vendor/three.min.js"
