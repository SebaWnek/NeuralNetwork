namespace NeuralNetwork
{
    public interface ILayer
    {
        void CalculateOutput(double[] previousLayer);
        void CalculateBackPropagationDeltas(double[] desiredOutput, Layer previousLayer);
        void CalculateBackPropagationDeltas(Layer nextLayer, Layer previousLayer);
        void CalculateBackPropagationDeltas(Layer nextLayer, Input input);
        void UpdateLayer();
    }
}