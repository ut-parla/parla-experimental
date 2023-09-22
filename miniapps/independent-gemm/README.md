We can't put this on the main repo, as our github repo contains the cublasMG binaries from NVIDIA. I'm fairly sure if we host on a public repo that would violate their liscense. 

The example is available at: https://github.com/ut-parla/cublasmg-wrapper

Requires:
- CUDA 10.1

Installation:
- make clean; make

This app runs independent gemms of different sizes. Data is moved through Crosspy (manual or automatic)
