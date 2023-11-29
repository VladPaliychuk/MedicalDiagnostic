﻿namespace MedicalDiagnostic
{
    public class Diagnostic
    {
        private int epochs;
        private double[] input;
        private double[,] trainingData;
        private double[] target;
        private double[,] targetMatrix;
        private Neuron[] hiddentNeurons;
        private Neuron outputNeuron;
        public Diagnostic(int epochs, double[,] trainingData, double[] target)
        {
            this.epochs = epochs;
            this.trainingData = trainingData;
            this.target = target;

            Train();
            Consultation();
            Diagnosis();
        }
        public Diagnostic(int epochs, double[,] trainingData, double[,] targetMatrix)
        {
            this.epochs = epochs;
            this.trainingData = trainingData;
            this.targetMatrix = targetMatrix;

            Train();
            Consultation();
            Diagnosis();
        }
        public void Consultation()
        {
            string[] symptoms = { "Головний біль", "Кашель", "Нежить", "Температура"};
            double[] consoleInput = new double[symptoms.Length];

            Console.WriteLine("Запишіть ваші симптоми: ");
            for (int i = 0; i < symptoms.Length; i++)
            {
                Console.WriteLine($"{symptoms[i]}: ");
                consoleInput[i] = int.Parse(Console.ReadLine());
            }
            input = consoleInput;
        }
        public void Diagnosis()
        {

             double[] hiddenOutput = new double[hiddentNeurons.Length];

             for (int n = 0; n < hiddentNeurons.Length; n++)
             {
                 hiddenOutput[n] = hiddentNeurons[n].CalculateOutput(input);
             }

             double output = outputNeuron.CalculateOutput(hiddenOutput);
             if(output >= 0.9d)
             {
                Console.WriteLine($"Ви хворієте грипом з шансом {output}%");
             }
            else
            {
                Console.WriteLine("Ви не хворієте грипом.");
            }
        }
        public void Train()
        {
            Initialize();
            for(int i = 0; i < epochs; i++)
            {
                Console.WriteLine("1");
                double[] outputs = new double[trainingData.GetLength(0)];
                for (int j = 0; j < trainingData.GetLength(0); j++)
                {
                    //Info();
                    double[] row = GetRow(trainingData, j);

                    double[] hiddenOutput = new double[hiddentNeurons.Length];

                    for(int n=0; n < hiddentNeurons.Length; n++)
                    {
                        hiddenOutput[n] = hiddentNeurons[n].CalculateOutput(row);
                    }

                    double output = outputNeuron.CalculateOutput(hiddenOutput);
                    outputNeuron.CalculateError(target[j], output);
                    outputNeuron.UpdateWeights(row);
                    outputNeuron.UpdateBiases();

                    for (int n = 0; n < hiddentNeurons.Length; n++)
                    {
                        hiddentNeurons[n].BackpropagateError(output, outputNeuron.Error);
                        hiddentNeurons[n].UpdateWeights(row);
                        hiddentNeurons[n].UpdateBiases();
                    }
                    outputs[j] += output;
                }
                if(Check(outputs) == true)
                {
                    break;
                }
            }
            for (int j = 0; j < trainingData.GetLength(0); j++)
            {
                double[] row = GetRow(trainingData, j);
                double[] hiddenOutput = new double[hiddentNeurons.Length];

                for (int n = 0; n < hiddentNeurons.Length; n++)
                {
                    hiddenOutput[n] = hiddentNeurons[n].CalculateOutput(row);
                }

                double output = outputNeuron.CalculateOutput(hiddenOutput);

                Console.WriteLine($"{output}");
            }
        }
        public void TrainSecond()
        {

        }
        public bool Check(double[] outputs)
        {
            int res = 0;
            for(int i = 0; i < target.Length; i++)
            {
                if ((target[i] == 0) && (outputs[i] <= 0.5d))
                {
                    res++;
                }
                if ((target[i] == 1) && (outputs[i] >= 0.9d))
                {
                    res++;
                }
            }

            if (res == target.Length)
            {
                return true;
            }
            return false;
        }
        public void Info()
        {
            foreach(var item in hiddentNeurons)
            {
                Console.WriteLine(item.ToString());
                foreach(var weight in item.Weights)
                {
                    Console.WriteLine(weight.ToString());
                }
                Console.WriteLine("\n");
            }

            Console.WriteLine(outputNeuron.ToString());
            foreach(var weight in outputNeuron.Weights)
            {
                Console.WriteLine(weight.ToString());
            }
            Console.WriteLine("\n");
        }
        public void Initialize()
        {
            hiddentNeurons = new Neuron[2];

            for (int n = 0; n < hiddentNeurons.Length; n++)
            {
                hiddentNeurons[n] = new Neuron(trainingData.GetLength(1));
            }

            outputNeuron = new Neuron(hiddentNeurons.Length);
        }
        static double[] GetRow(double[,] matrix, int rowIndex)
        {
            int cols = matrix.GetLength(1);
            double[] row = new double[cols];

            for (int i = 0; i < cols; i++)
            {
                row[i] = matrix[rowIndex, i];
            }

            return row;
        }
    }
}
