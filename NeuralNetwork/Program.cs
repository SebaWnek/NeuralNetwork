using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Network network = new Network(4, 3, 3, 2);
            network.CalculateNetwork(new float[] { 3, 2, 4, 1 });
            float[] result = network.GetOutputs();
            foreach (float i in result) Console.WriteLine(i);
        }
    }

    class Input
    {
        public float[] Inputs { get; set; }

        public Input(int count)
        {
            Inputs = new float[count];
        }
    }

    class Layer
    {
        static readonly int minRand = -1;
        static readonly int maxRand = 1;
        static readonly private float learningRate = 0.0001f;
        static readonly Random random = new Random();
        Func<float[], float[]> Function;
        Func<float[], float[]> Derivative;

        public float[,] Weights { get; set; }
        public float[,] DWeights { get; set; }
        public float[] Biases { get; set; }
        public float[] Outputs { get; set; }
        public float[] Gammas { get; set; }

        public Layer(int prevCount, int count)
        {
            Biases = new float[count];
            Outputs = new float[count];
            Gammas = new float[count];
            Weights = new float[count, prevCount];
            DWeights = new float[count, prevCount];
            InitializeLayer();
            Function = TanH;
            Derivative = DTanH;
        }

        private void InitializeLayer()
        {
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = GetRandom();
            }
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    Weights[i, j] = GetRandom();
                }
            }
        }

        private static float GetRandom()
        {
            return (float)random.Next(100 * minRand, 100 * maxRand) / 100;
        }

        public void CalculateOutput(float[] previousLayer)
        {
            MultiplyWeightsInputs(previousLayer);
            AddBias();
            Outputs = Function(Outputs);
        }

        private void MultiplyWeightsInputs(float[] previousLayer)
        {
            Outputs = new float[Outputs.Length]; //clear outputs
            for (int i = 0; i < Outputs.Length; i++)
            {
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    Outputs[i] += previousLayer[j] * Weights[i, j];
                }
            }
        }

        private void AddBias()
        {
            for (int i = 0; i < Outputs.Length; i++)
            {
                Outputs[i] += Biases[i];
            }
        }

        private float[] TanH(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (float)Math.Tanh(input[i]);
            }
            return output;
        }

        private float[] DTanH(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1 - input[i] * input[i];
            }
            return output;
        }

        public void CalculateGammas()
        {

        }
        public void CalculateOutputGammas(float[] desiredOutputs)
        {
            float[] derivatives = Derivative(Outputs);
            for(int i = 0; i < Gammas.Length; i++)
            {
                Gammas[i] = (Outputs[i] - desiredOutputs[i]) * derivatives[i];
            }
        }

        public void CalculateOutputWeightsDerivatives(float[] previousOutputs)
        {
            for (int i = 0; i < Gammas.Length; i++)
            {
                for (int j = 0; j < previousOutputs.Length; j++)
                {
                    DWeights[i, j] = Gammas[i] * previousOutputs[j];
                }
            }
        }
    }

    class Network
    {
        public Input Input { get; set; }
        public Layer[] Layers { get; set; }

        public Network(params int[] layers)
        {
            Layers = new Layer[layers.Length - 1];
            if (layers.Length >= 2)
            {
                Input = new Input(layers[0]);
                for (int i = 1; i < layers.Length; i++)
                {
                    Layers[i - 1] = new Layer(layers[i - 1], layers[i]);
                }
            }
            else throw new ArgumentException("Not enough layers!");
        }

        public float[] GetOutputs()
        {
            return Layers[Layers.Length - 1].Outputs;
        }

        public void CalculateNetwork(float[] inputs)
        {
            if (inputs.Length == Input.Inputs.Length)
            {
                Input.Inputs = inputs; 
            }
            else
            {
                Console.WriteLine("Wrong input!");
                return;
            }

            Layers[0].CalculateOutput(Input.Inputs);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].CalculateOutput(Layers[i - 1].Outputs);
            }
        }

        private void CalculateBackPropagation(float[] inputs, float[] desiredOutput)
        {
            if (inputs.Length == Input.Inputs.Length && desiredOutput.Length == GetOutputs().Length)
            {
                Input.Inputs = inputs;
            }
            else
            {
                Console.WriteLine("Wrong input or desired output!");
                return;
            }
            Layers[Layers.Length - 1].CalculateOutputGammas(desiredOutput);
            Layers[Layers.Length - 1].CalculateOutputWeightsDerivatives(Layers[Layers.Length - 2].Outputs);
        }

        public void TrainNetwork(float[] inputs, float[] desiredOutput)
        {
            CalculateNetwork(inputs);
            CalculateBackPropagation(inputs, desiredOutput);
        }
    }
}
