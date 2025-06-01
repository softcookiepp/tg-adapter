# tg-adapter
Adapter library for porting pytorch code to tinygrad

## Installation
Since there is no pypi package, it must (for now) be installed by running:
`pip install -U git+https://github.com/softcookiepp/tg-adapter.git`

## Testing
Tests are still being migrated from tiny-hf, they cannot be run yet

## Usage
### API
The API aims to be a drop-in replacement for pytorch.
Simply replace every `import torch` with `import tg_adapter as torch` in your code, and it should likely run with minimal changes required.

### Environment variables
A lot of pytorch code is tied directly to CUDA explicitly, via
`torch.Tensor.cuda()`, etc.
Explicit handling functions can be overridden with the tinygrad backend
of the user's choice, by assigning the vairable `TGA_CUDA_OVERRIDE`.
For example, `TGA_CUDA_OVERRIDE=GPU` or `TGA_CUDA_OVERRIDE=WEBGPU`, etc.
