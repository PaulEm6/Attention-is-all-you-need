import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

print(enc.encode("hello world"))
print(enc.decode(enc.encode("hello world")))


# To get the tokeniser corresponding to a specific model in the OpenAI API:
#enc = tiktoken.encoding_for_model("gpt-4")