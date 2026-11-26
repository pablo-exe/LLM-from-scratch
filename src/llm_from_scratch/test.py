import tiktoken
import torch

from .model import GPTModel

model = GPTModel(768, 12, 12, 50257, 0.1, 1024)

test = 'hola, me llamo'
tokenizer = tiktoken.get_encoding("gpt2")

input = tokenizer.encode(test)
input_tensor = torch.tensor(input).unsqueeze(0)
print(input_tensor, input_tensor.shape)

model.eval()

output = model.forward(input_tensor)
print(output.shape)
next_token_logits = output[:, -1, :]
prediction = torch.argmax(next_token_logits, dim=-1)

print(test, tokenizer.decode([int(prediction)]))