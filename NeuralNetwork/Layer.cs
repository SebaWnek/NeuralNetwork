using System;

namespace NeuralNetwork
{
    public class Layer
    {
        static readonly int minRand = -1;
        static readonly int maxRand = 1;
        static readonly private double learningRate = 0.0001f;
        static readonly Random random = new Random();
        public Func<double[], double[]> Function { get; set; }
        public Func<double[], double[]> Derivative { get; set; }

        public double[,] Weights { get; set; }
        public double[,] DWeights { get; set; }
        public double[] Biases { get; set; }
        public double[] DBiases { get; set; }
        public double[] Outputs { get; set; }
        public double[] Gammas { get; set; }

        public Layer(int prevCount, int count)
        {
            Biases = new double[count];
            DBiases = new double[count];
            Outputs = new double[count];
            Gammas = new double[count];
            Weights = new double[count, prevCount];
            DWeights = new double[count, prevCount];
            InitializeLayer();
            Function = Functions.TanH;
            Derivative = Functions.DTanH;
        }

        private void InitializeLayer()
        {
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = 5 * GetRandom();
            }
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    Weights[i, j] = GetRandom();
                }
            }
        }

        private static double GetRandom()
        {
            return (double)random.Next(100 * minRand, 100 * maxRand) / 100;
        }

        public void CalculateOutput(double[] previousLayer)
        {
            MultiplyWeightsInputs(previousLayer);
            AddBias();
            Outputs = Function(Outputs);
        }

        private void MultiplyWeightsInputs(double[] previousLayer)
        {
            Outputs = new double[Outputs.Length]; //clear outputs
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
            Gammas = new double[Gammas.Length]; //clear gammas
            double[] derivatives = Derivative(Outputs);
            for (int i = 0; i < Gammas.Length; i++)
            {
                for (int j = 0; j < nextLayer.Gammas.Length; j++)
                {
                    Gammas[i] += nextLayer.Gammas[j] * nextLayer.Weights[j, i];
                }
                Gammas[i] *= derivatives[i];
            }
        }

        private void CalculateGammas(double[] desiredOutputs)
        {
            double[] derivatives = Derivative(Outputs);
            for (int i = 0; i < Gammas.Length; i++)
            {
                Gammas[i] = (desiredOutputs[i] - Outputs[i]) * derivatives[i];
            }
        }

        private void CalculateWeightsDerivatives(double[] previousOutputs)
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
                    Weights[i, j] += learningRate * DWeights[i, j];
                }
            }
        }

        private void UpdateBiases()
        {
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] += learningRate * DBiases[i];
            }
        }
        /// <summary>
        /// Calculates output layer
        /// </summary>
        /// <param name="desiredOutput">Desired outputs</param>
        /// <param name="previousLayer">Preceeding layer</param>
        public void CalculateBackPropagationDeltas(double[] desiredOutput, Layer previousLayer)
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
