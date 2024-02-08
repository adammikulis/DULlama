This is the first iteration of a bare-bones local RAG pipeline in C# using the LLamaSharp nuget package. The program constructs a basic vector database consisting of the original text and the embeddings generated by the chosen LLM. The user submits a query, which gets matched against the db using cosine similarity. This returns the top results (original text) which are then integrated into the LLM's response. This has been tested and immediately corrects wrong information generated by an LLM by itself without RAG.

All Llama2 and Mistral 7B models require a mininum of 16GB of RAM for CPU inference (8GB for GPU if using a CUDA backend). 
Add one of the nuget packages listed here using Nuget Package Manager to use an Nvidia GPU (you may have to remove the CPU backend):
https://www.nuget.org/packages/LLamaSharp.Backend.Cuda11
https://www.nuget.org/packages/LLamaSharp.Backend.Cuda12

Mistral model downloads:
Math-enhanced: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-DARE-GGUF
Coding: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF
General instruct: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF 

Llama2 models:
Coding: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF
General: https://huggingface.co/TheBloke/Llama-2-7B-GGUF

Inference speed on CPU is roughly equal between Mistral and Llama2, with Mistral scoring higher in benchmarks. The next update will support Microsoft's Phi-2 which will require half of the current system requirements.

Future steps:
- Allow the LLM to process the entire conversation while still focusing on the most recent prompt to avoid repeating answers
- Integrate some telemetry like tok/s
- Build a better UI
