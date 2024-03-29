# blip2_8bit
BLIP-2 Captioning with 8-bit Quantization. \
I made this before HuggingFace had integrated the BLIP-2 model. Probably better to use [their implementation](https://huggingface.co/docs/transformers/main/model_doc/blip-2) now, which supports their 8-bit quantization.

# REQUIREMENTS
[accelerate](https://github.com/huggingface/accelerate) \
[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) \
[LAVIS](https://github.com/salesforce/LAVIS) \
[Pillow](https://github.com/python-pillow/Pillow) \
[torch](https://pytorch.org/)

I've been using this to install bitsandbytes for the current version of CUDA (XX.X): \
`pip install -i https://test.pypi.org/simple/ bitsandbytes-cudaXXX`

# USAGE
Using this to run BLIP-2 with the OPT6.7b captioning model on 12GB of VRAM.
