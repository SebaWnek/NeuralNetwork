using System;

namespace NeuralNetwork
{
    public class Network
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

        public double[] GetOutputs()
        {
            return Layers[Layers.Length - 1].Outputs;
        }

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

        public void TrainNetwork(double[] inputs, double[] desiredOutput)
        {
            CalculateNetwork(inputs);
            CalculateBackPropagation(desiredOutput);
        }
    }
}
