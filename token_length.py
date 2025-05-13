import os
import tiktoken

enc = tiktoken.get_encoding("gpt2")
toks = []
# Open every file in the directory and count the tokens
for file in os.listdir("exemples_MedMCQA"):
    if file.endswith(".txt"):
        with open(os.path.join("exemples_MedMCQA", file)) as f:
            text = f.read()
        tokens = enc.encode(text)
        toks.append((file, len(tokens)))


# Sort the tokens in descending order
toks.sort(key=lambda x: x[1], reverse=True)

# Print the tokens in descending order
for file, num_tokens in toks:
    print(f"{file}: {num_tokens} tokens")
