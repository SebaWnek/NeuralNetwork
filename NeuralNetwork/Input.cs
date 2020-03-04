namespace NeuralNetwork
{
    /// <summary>
    /// Network first layer containing input data
    /// </summary>
    public class Input
    {
        public double[] Inputs { get; set; }

        public Input(int count)
        {
            Inputs = new double[count];
        }
    }
}
