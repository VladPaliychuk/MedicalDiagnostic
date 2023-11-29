using MedicalDiagnostic;

Console.OutputEncoding = System.Text.Encoding.Unicode;

int epochs;

double[,] trainingData = { 
    { 0, 0, 1, 1 }, 
    { 1, 0, 0, 0 }, 
    { 1, 1, 1, 1 }, 
    { 0, 0, 0, 1 },
    { 0, 0, 0, 0 },
    { 1, 0, 1, 1 },
};
double[] targets = { 1, 0, 1, 0, 0, 1};
double[,] targetMatrix =
{
    {0, 1, 0 },
    {1, 0, 0 },
    {1, 1, 0 },
    {0, 0, 1 },
    {0, 0, 1 },
    {1, 1, 0 }
};

Console.WriteLine("Оберіть діагностику 1/2 \n(1 - один варіант відповіді, 2 - три варіанти відповіді)");
int c = int.Parse(Console.ReadLine());

Console.WriteLine("Кількість епох: ");
epochs = int.Parse(Console.ReadLine());

switch (c)
{
    case 1:
        Diagnostic diagnostic1 = new Diagnostic(epochs, trainingData, targets);
        break;
    case 2:
        Diagnostic diagnostic2 = new Diagnostic(epochs, trainingData, targetMatrix);
        break;
}
