using System;

namespace NeuralNetwork
{
    /// <summary>
    /// Hidden and output layers
    /// </summary>
    public class Layer
    {
        static readonly int minRand = -1; //Allowing to change random numbers generator behavoiur
        static readonly int maxRand = 1; //Allowing to change random numbers generator behavoiur
        static readonly int randDivisor = 100; //Allowing to change random numbers generator behavoiur
        static readonly Random random = new Random();
        public Func<double[], double[]> Function { get; set; }
        public Func<double[], double[]> Derivative { get; set; }

        public double[,] Weights { get; set; }
        public double[,] DWeights { get; set; }
        public double[] Biases { get; set; }
        public double[] DBiases { get; set; }
        public double[] Outputs { get; set; }
        public double[] Gammas { get; set; }

        public double LearningRate { get; set; } = 0.00005;
        public double BiasMultiplier { get; set; } = 10;
        public double WeightsMultiplier { get; set; } = 2;

        /// <summary>
        /// Main constructor for layer
        /// </summary>
        /// <param name="prevCount">Count of nodes in preceeding layer</param>
        /// <param name="count">Count of nodes in current layer</param>
        /// <param name="type">Activation function type</param>
        /// <param name="learningRate">Learning rate</param>
        /// <param name="biasMultip">Range of starting biases - random number between -biasMultip and biasMultip</param>
        /// <param name="weightsMultip">Range of starting biases - random number between -weihtsMultip and weightsMultip</param>
        public Layer(int prevCount, int count, FunctionTypes type = FunctionTypes.Sigmoid, double learningRate = 0.00005, double biasMultip = 10, double weightsMultip = 2)
        {
            Biases = new double[count];
            DBiases = new double[count];
            Outputs = new double[count];
            Gammas = new double[count];
            Weights = new double[count, prevCount];
            DWeights = new double[count, prevCount];
            LearningRate = learningRate;
            BiasMultiplier = biasMultip;
            WeightsMultiplier = weightsMultip;
            (Function, Derivative) = Functions.GetFunctions(type);
            InitializeLayer();
        }

        /// <summary>
        /// Generating random biases and weights
        /// </summary>
        private void InitializeLayer()
        {
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] = BiasMultiplier * GetRandom();
            }
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    Weights[i, j] = WeightsMultiplier * GetRandom();
                }
            }
        }

        private static double GetRandom()
        {
            return (double)random.Next(100 * minRand, 100 * maxRand) / randDivisor;
        }

        /// <summary>
        /// Calculating layer's output values
        /// </summary>
        /// <param name="previousLayer">Layer's input values provided by preceeding layer</param>
        public void CalculateOutput(double[] previousLayer)
        {
            MultiplyWeightsInputs(previousLayer);
            AddBias();
            Outputs = Function(Outputs);
        }

        /// <summary>
        /// Series of helper methods for calculating different steps of output calculation and backpropagation
        /// </summary>
        
        #region Helper methods
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
                    Weights[i, j] += LearningRate * DWeights[i, j];
                }
            }
        }

        private void UpdateBiases()
        {
            for (int i = 0; i < Biases.Length; i++)
            {
                Biases[i] += LearningRate * DBiases[i];
            }
        }
        #endregion

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
        /// <summary>
        /// Update nodes' parameters with newly calculated differences
        /// </summary>
        public void UpdateLayer()
        {
            UpdateWeights();
            UpdateBiases();
        }
    }
}
