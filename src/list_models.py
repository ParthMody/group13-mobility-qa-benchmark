import os


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY not set.  set GEMINI_API_KEY=your-key")
    from google import genai
    client = genai.Client(api_key=key)
    print("Models your key can see (name | supported actions):\n")
    for m in client.models.list():
        actions = (getattr(m, "supported_actions", None)
                   or getattr(m, "supported_generation_methods", None) or "?")
        name = getattr(m, "name", str(m))
        # only show the ones useful for us
        if "generateContent" in str(actions) or actions == "?":
            print(f"  {name:45} | {actions}")
    print("\nUse one of the above with:  python src/llm_zeroshot.py --model <name>")
    print("(drop any 'models/' prefix — e.g. use 'gemini-2.5-flash')")


if __name__ == "__main__":
    main()