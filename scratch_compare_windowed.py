import urllib.request
import os

original_url = "https://raw.githubusercontent.com/Vchitect/TRELLIS/main/trellis/modules/sparse/attention/windowed_attn.py"
urllib.request.urlretrieve(original_url, "windowed_attn_orig.py")
