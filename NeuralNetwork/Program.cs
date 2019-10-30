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
            Random random = new Random();
            Network network = new Network(2, 2, 1);
            //network.Layers[network.Layers.Length - 1].Function = Functions.LeakyReLU;
            //network.Layers[network.Layers.Length - 1].Derivative = Functions.DLeakyReLU;
            for (int i = 0; i < 1000000; i++)
            {
                float[] inputs = new float[] { random.Next(11), random.Next(1, 11) };
                int res = (int)(inputs[0] + inputs[1]);
                float[] output = new float[1];
                output[0] = res;
                //Console.WriteLine(inputs[0] + " * " + inputs[1] + " = " + res);
                //foreach (float j in output) Console.Write(j + ", ");
                //Console.WriteLine();
                network.TrainNetwork(inputs, output);
            }
            for (int i = 0; i < 10; i++)
            {
                float[] input = new float[] { random.Next(11), random.Next(11) };
                network.CalculateNetwork(input);
                float[] result = network.GetOutputs();
                Console.Write(input[0] + " + " + input[1] + " = ");
                foreach (float j in result) Console.Write((j) + " => ");
                Console.WriteLine((int)Math.Round(result[0]) == (input[0] + input[1]));
                Console.WriteLine();
            }
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

    static class Functions
    {
        public static float[] TanH(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (float)Math.Tanh(input[i]);
            }
            return output;
        }

        public static float[] DTanH(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = 1 - input[i] * input[i];
            }
            return output;
        }

        public static float[] LeakyReLU(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? input[i] : 0.01f * input[i];
            }
            return output;
        }

        public static float[] DLeakyReLU(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] >= 0 ? 1 : 0.01f;
            }
            return output;
        }

        public static float[] Sigmoid(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = (float)(1 / (1 + Math.Pow(Math.E, -1 * input[i])));
            }
            return output;
        }

        public static float[] DSigmoid(float[] input)
        {
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = input[i] * (1 - input[i]);
            }
            return output;
        }
    }

    class Layer
    {
        static readonly int minRand = 0;
        static readonly int maxRand = 1;
        static readonly private float learningRate = 0.001f;
        static readonly Random random = new Random();
        public Func<float[], float[]> Function { get; set; }
        public Func<float[], float[]> Derivative { get; set; }

        public float[,] Weights { get; set; }
        public float[,] DWeights { get; set; }
        public float[] Biases { get; set; }
        public float[] DBiases { get; set; }
        public float[] Outputs { get; set; }
        public float[] Gammas { get; set; }

        public Layer(int prevCount, int count)
        {
            Biases = new float[count];
            DBiases = new float[count];
            Outputs = new float[count];
            Gammas = new float[count];
            Weights = new float[count, prevCount];
            DWeights = new float[count, prevCount];
            InitializeLayer();
            Function = Functions.LeakyReLU;
            Derivative = Functions.DLeakyReLU;
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

        private void CalculateBiasDerivatives()
        {
            for (int i = 0; i < DBiases.Length; i++)
            {
                DBiases[i] = Gammas[i];
            }
        }

        private void CalculateGammas(Layer nextLayer)
        {
            float[] gamma = new float[Gammas.Length];
            float[] derivatives = Derivative(Outputs);
            for (int i = 0; i < Gammas.Length; i++)
            {
                for (int j = 0; j < nextLayer.Gammas.Length; j++)
                {
                    gamma[i] += nextLayer.Gammas[j] * nextLayer.Weights[j, i];
                }
                gamma[i] *= derivatives[i];
            }
        }

        private void CalculateGammas(float[] desiredOutputs)
        {
            float[] derivatives = Derivative(Outputs);
            for (int i = 0; i < Gammas.Length; i++)
            {
                Gammas[i] = (Outputs[i] - desiredOutputs[i]) * derivatives[i];
            }
        }

        private void CalculateWeightsDerivatives(float[] previousOutputs)
        {
            for (int i = 0; i < Gammas.Length; i++)
            {
                for (int j = 0; j < previousOutputs.Length; j++)
                {
                    DWeights[i, j] = Gammas[i] * previousOutputs[j];
                }
            }
        }

        private void UpdateWeights()
        {
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    Weights[i, j] -= learningRate * DWeights[i, j];
                }
            }
        }

        private void UpdateBiases()
        {
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] -= learningRate * DBiases[i];
            }
        }
        /// <summary>
        /// Calculates output layer
        /// </summary>
        /// <param name="desiredOutput">Desired outputs</param>
        /// <param name="previousLayer">Preceeding layer</param>
        public void CalculateBackPropagationDeltas(float[] desiredOutput, Layer previousLayer)
        {
            CalculateGammas(desiredOutput);
            CalculateWeightsDerivatives(previousLayer.Outputs);
            CalculateBiasDerivatives();
        }
        /// <summary>
        /// Calculates hidden layers except for first one
        /// </summary>
        /// <param name="nextLayer">Following layer</param>
        /// <param name="previousLayer">Preceeding layer</param>
        public void CalculateBackPropagationDeltas(Layer nextLayer, Layer previousLayer)
        {
            CalculateGammas(nextLayer);
            CalculateWeightsDerivatives(previousLayer.Outputs);
            CalculateBiasDerivatives();
        }
        /// <summary>
        /// Calculates first hidden layer, next after input layer
        /// </summary>
        /// <param name="nextLayer">Following layer</param>
        /// <param name="input">Network inputs</param>
        public void CalculateBackPropagationDeltas(Layer nextLayer, Input input)
        {
            CalculateGammas(nextLayer);
            CalculateWeightsDerivatives(input.Inputs);
            CalculateBiasDerivatives();
        }

        public void UpdateLayer()
        {
            UpdateWeights();
            UpdateBiases();
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

        private void CalculateBackPropagation(float[] desiredOutput)
        {
            for(int i = Layers.Length - 1; i >= 0; i--)
            {
                if (i == Layers.Length - 1)
                {
                    Layers[i].CalculateBackPropagationDeltas(desiredOutput, Layers[i - 1]);
                }
                else if (i > 0)
                {
                    Layers[i].CalculateBackPropagationDeltas(Layers[i + 1], Layers[i - 1]);
                }
                else
                {
                    Layers[i].CalculateBackPropagationDeltas(Layers[i + 1], Input);
                }
            }
            for(int i = Layers.Length - 1; i >= 0; i--)
            {
                Layers[i].UpdateLayer();
            }
        }

        public void TrainNetwork(float[] inputs, float[] desiredOutput)
        {
            CalculateNetwork(inputs);
            CalculateBackPropagation(desiredOutput);
        }
    }
}
