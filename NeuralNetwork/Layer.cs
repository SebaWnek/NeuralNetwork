using System;

namespace NeuralNetwork
{
    class Layer
    {
        static readonly int minRand = -1;
        static readonly int maxRand = 1;
        static readonly private float learningRate = 0.000001f;
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
            Function = Functions.Linear;
            Derivative = Functions.DLinear;
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
}
