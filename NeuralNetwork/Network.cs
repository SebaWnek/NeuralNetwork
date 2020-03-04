using System;

namespace NeuralNetwork
{
    public class Network : INeuralNetwork
    {
        public Input Input { get; set; }
        public Layer[] Layers { get; set; }

        /// <summary>
        /// Constructor to be used only when layers parameers are set up in the code.
        /// </summary>
        /// <param name="layers">Number of nodes in consecutive layers. Must be >= 2</param>
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

        /// <summary>
        /// Main constructor for network
        /// </summary>
        /// <param name="layers">Number of nodes in consecutive layers. Must be >= 2</param>
        /// <param name="baseType">Type of activation function in hidden layers</param>
        /// <param name="outputType">Type of activation function in output layer</param>
        /// <param name="learningRate">Learning rate</param>
        /// <param name="biasMultip">Range of starting biases - random number between -biasMultip and biasMultip</param>
        /// <param name="weightsMultip">Range of starting biases - random number between -weihtsMultip and weightsMultip</param>
        public Network(int[] layers, FunctionTypes baseType, FunctionTypes outputType, double learningRate, double biasMultip, double weightsMultip) 
            : this(layers)
        {
            Layers = new Layer[layers.Length - 1];
            if (layers.Length >= 2)
            {
                Input = new Input(layers[0]);
                for (int i = 1; i < layers.Length - 1; i++)
                {
                    Layers[i - 1] = new Layer(layers[i - 1], layers[i], baseType, learningRate, biasMultip, weightsMultip);
                }
                Layers[Layers.Length - 1] = new Layer(layers[layers.Length - 2], layers[layers.Length - 1], outputType, learningRate, biasMultip, weightsMultip);
            }
            else throw new ArgumentException("Not enough layers!");
        }

        /// <summary>
        /// Public method for getting output from calculated method. 
        /// </summary>
        /// <returns>Array of results</returns>
        public double[] GetOutputs()
        {
            return Layers[Layers.Length - 1].Outputs;
        }

        /// <summary>
        /// Calculate network with provided inputs
        /// </summary>
        /// <param name="inputs">Input array to be passed through network</param>
        public void CalculateNetwork(double[] inputs)
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

        /// <summary>
        /// Calculating backpropagation algorythm on calculated network. Not to be used on it's own
        /// </summary>
        /// <param name="desiredOutput">Expected results for backpropatagtion calculation</param>
        private void CalculateBackPropagation(double[] desiredOutput)
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

        /// <summary>
        /// Training method
        /// </summary>
        /// <param name="inputs">Training data</param>
        /// <param name="desiredOutput">Expected results</param>
        public void TrainNetwork(double[] inputs, double[] desiredOutput)
        {
            CalculateNetwork(inputs);
            CalculateBackPropagation(desiredOutput);
        }

        /// <summary>
        /// Allows changing learning rate suring training process
        /// </summary>
        /// <param name="learningRate">New learning rate</param>
        public void UpdateLearningRate(double learningRate)
        {
            foreach (Layer layer in Layers) layer.LearningRate = learningRate;
        }
    }
}
