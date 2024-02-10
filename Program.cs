// This program creates a local RAG pipeline using LLamaSharp v0.9.1 utilizing Mistral or Llama2 transformer models
// It has a model directoryPath set to C:\ai\models, change to where you keep your downloaded models

// Mistral math-enhanced model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-DARE-GGUF
// Mistral coding model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-code-ft-GGUF
// Mistral general instruct model: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
// Llama2 coding model: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF
// Llama2 general model: https://huggingface.co/TheBloke/Llama-2-7B-GGUF

// This project includes LLamaSharp.Backend.CPU only, for Nvidia GPUs add either backend using Nuget Package Manager in Visual Studio or via VSCode extension:
// https://www.nuget.org/packages/LLamaSharp.Backend.Cuda11
// https://www.nuget.org/packages/LLamaSharp.Backend.Cuda12

using System;
using System.Threading.Tasks;

class Program
{
    static async Task Main(string[] args)
    {
        string directoryPath = @"C:\ai\models";
        string[] facts = new string[] {
            "The University of Denver is a private University that is abbreviated as 'DU'",
            "The University of Denver was founded in 1864",
            "DU is a private R1 University",
            "The mascot of the University of Denver is the Pioneer",
            "DU is located in south Denver, Colorado in the University neighborhood",
            "DU's has a secondary/satellite campus, the 720 acre Kennedy Mountain Campus which is located 110 miles northwest of Denver",
            "DU has 5700 undergraduate students and 7200 graduate students",
            "DU's Ritchie Center is home to the Magness Arena",
            "DU's hockey team plays in Magness Arena, named after cable television pioneer Bob Magness",
            "The Pioneers won the ice hockey NCAA National Championship in 2022"
        };
        uint contextSize = 4096;

        var pipeline = new RagPipelineConsole(directoryPath, facts, contextSize);
        await pipeline.InitializeAsync();
        await pipeline.StartChatAsync();
    }
}
