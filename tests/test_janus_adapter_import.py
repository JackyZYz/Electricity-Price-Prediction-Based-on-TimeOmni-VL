from timeomni_vl.models import build_adapter

try:
    adapter = build_adapter("janus")
    print(f"Built adapter: {type(adapter).__name__}")
    print("JanusAdapter can be imported successfully on CPU.")
    print("NOTE: load() requires CUDA and Janus package installed from source.")
except Exception as e:
    print(f"Error: {e}")
