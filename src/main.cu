#define IMPLEMENTATION

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
#include <iostream>
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::string errorMessage = "GPU Error: " + std::string(cudaGetErrorString(code)) +
                                   " in file " + std::string(file) +
                                   " at line " + std::to_string(line);
        throw std::runtime_error(errorMessage);
    }
}

#include "gpu_kernel_alg.cuh"
#include "gpu_thrust_alg.cuh"

#include "cpu_alg.h"
#include "file_manager.h"

template <int d>
void executeKMeansAlgorithm(const char *computation_method, const char *data_format, FILE *file, float **&points, int N, int k, const char *filename_output);

int main(int argc, char **argv)
{
    // Weryfikacja parametrów podanych przez użytkownika
    if (argc != 5)
    {
        printf("Niepoprawna ilość argumentów wywołania funkcji! Wywołanie musi być następujące:\n KMeans data_format computation_method input_file output_file\n");
        return 1;
    }

    const char *data_format = argv[1];
    if ((strcmp(data_format, "txt") != 0) && (strcmp(data_format, "bin") != 0))
    {
        printf("Niepoprawny format danych. Poprawne formaty danych: txt, bin. Wczytany format danyc: %s\n", data_format);
        return 1;
    }

    const char *computation_method = argv[2];
    if ((strcmp(computation_method, "gpu1") != 0) && (strcmp(computation_method, "gpu2") != 0) && (strcmp(computation_method, "cpu") != 0))
    {
        printf("Niepoprawna metoda wykonania. Poprawne metody: gpu1, gpu2, cpu. Wczytana metoda: %s\n", computation_method);
        return 1;
    }

    const char *filename_input = argv[3];
    FILE *file = fopen(filename_input, "r");
    if (file == NULL)
    {
        printf("Podany plik do odczytu nie jest poprawny\n");
        return 1;
    }

    const char *filename_output = argv[4];
    FILE *fileOutput = fopen(filename_output, "w");
    if (fileOutput == NULL)
    {
        perror("Podany plik do zapisu nie jest poprawny\n");
        return 1;
    }
    fclose(fileOutput);

    int N, d, k;
    float **points = NULL;

    // Wczytanie wymiarów danych (parametrów N, d, k)
    if (strcmp(data_format, "txt") == 0)
    {
        readHeaderFromTXT(file, &N, &d, &k);
    }
    else
    {
        readHeaderFromBIN(file, &N, &d, &k);
    }

    printf("Wczytano parametry: N=%d, d=%d, k=%d. Podany format danych: %s\nWykorzystywany algorytm obliczeniowy: %s\n", N, d, k, data_format, computation_method);

    // Wczytanie punktów oraz wywołanie odpowiedniej funkcji w zależności od podanej metody: cpu, gpu1, gpu2
    switch (d)
    {
    case 1:
        executeKMeansAlgorithm<1>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 2:
        executeKMeansAlgorithm<2>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 3:
        executeKMeansAlgorithm<3>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 4:
        executeKMeansAlgorithm<4>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 5:
        executeKMeansAlgorithm<5>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 6:
        executeKMeansAlgorithm<6>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 7:
        executeKMeansAlgorithm<7>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 8:
        executeKMeansAlgorithm<8>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 9:
        executeKMeansAlgorithm<9>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 10:
        executeKMeansAlgorithm<10>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 11:
        executeKMeansAlgorithm<11>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 12:
        executeKMeansAlgorithm<12>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 13:
        executeKMeansAlgorithm<13>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 14:
        executeKMeansAlgorithm<14>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 15:
        executeKMeansAlgorithm<15>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 16:
        executeKMeansAlgorithm<16>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 17:
        executeKMeansAlgorithm<17>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 18:
        executeKMeansAlgorithm<18>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 19:
        executeKMeansAlgorithm<19>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    case 20:
        executeKMeansAlgorithm<20>(computation_method, data_format, file, points, N, k, filename_output);
        break;
    default:
        std::cerr << "Unsupported dimension d: " << d << std::endl;
        break;
    }

    return 0;
}

// Funkcja odpowiedzialna za wybór odpowiedniej funkcji w załeżności od podanej metodu: cpu, gpu1, gpu2
template <int d>
void executeKMeansAlgorithm(const char *computation_method, const char *data_format, FILE *file, float **&points, int N, int k, const char *filename_output)
{
    // Wczytanie punktów do tablicy points
    readPointsFromFile<d>(data_format, file, N, points);

    if (strcmp(computation_method, "cpu") == 0)
    {
        KMeansCPU<d>(points, N, k, filename_output);
    }
    else if (strcmp(computation_method, "gpu1") == 0)
    {
        KMeansGPU1<d>(points, N, k, filename_output);
    }
    else if (strcmp(computation_method, "gpu2") == 0)
    {
        KMeansGPU2<d>(points, N, k, filename_output);
    }
}
