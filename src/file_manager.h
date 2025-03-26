#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

void readHeaderFromTXT(FILE *file, int *NP, int *dP, int *kP);

// template <int d>
// void readPointsFromTXT(FILE *file, int N, float **&points);

void readHeaderFromBIN(FILE *file, int *NP, int *dP, int *kP);

// template <int d>
// void readPointsFromBIN(FILE *file, int N, float **&points);

// template <int d>
// void readPointsFromFile(const char *data_format, FILE *file, int N, float **&points);

// template <int d>
// void writeToFile(const char *filename_output, int N, int k, float **centroids, int *nearestCentroidID);

// template <int d>
// void writeToFile(const char *filename_output, int N, int k, float *centroids, int *nearestCentroidID);


// Wczytanie parametrów N, d, k dla formatu TXT
// void readHeaderFromTXT(FILE *file, int *NP, int *dP, int *kP)
// {
//     fscanf(file, "%d %d %d", NP, dP, kP);
// }

// Wczytanie punktów dla formatu TXT
template <int d>
void readPointsFromTXT(FILE *file, int N, float **&points)
{
    printf("Wczytywanie punktów z pliku..\n");
    // Zapianie czasu początkowego wczytywania danych
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    size_t bytes_read;
    points = new float *[d];
    if (!points)
    {
        printf("Malloc error in function readPointsFromTXT with points\n");
        exit(1);
    }
    for (int i = 0; i < d; i++)
    {
        points[i] = new float[N];
        if (!points[i])
        {
            for (int j = 0; j < i; i++)
            {
                delete[] points[i];
            }
            delete[] points;
            printf("Malloc error in function readPointsFromTXT with points[%d]\n", i);
            exit(1);
        }
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < d; j++)
        {
            bytes_read = fscanf(file, "%f", &points[j][i]);
            if (bytes_read != 1)
            {
                printf("Fread error in function readPointsFromTXT with points[%d][%d]\n.", j, i);
                exit(1);
            }
        }
    }
    fclose(file);

    // Zapisanie czasu końcowego wczytywania danych i różnicy między start a stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Czas wykonania: %f milisekund\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Wczytanie parametrów N, d, k dla formatu BIN
// void readHeaderFromBIN(FILE *file, int *NP, int *dP, int *kP)
// {
//     size_t bytes_read = fread(NP, sizeof(int), 1, file);
//     if (bytes_read != 1)
//     {
//         printf("Fread error in function readHeaderFromBIN with NP\n.");
//         exit(1);
//     }

//     bytes_read = fread(dP, sizeof(int), 1, file);
//     if (bytes_read != 1)
//     {
//         printf("Fread error in function readHeaderFromBIN with dP\n.");
//         exit(1);
//     }

//     bytes_read = fread(kP, sizeof(int), 1, file);
//     if (bytes_read != 1)
//     {
//         printf("Fread error in function readHeaderFromBIN with kP\n.");
//         exit(1);
//     }
// }

// Wczytanie punktów dla formatu BIN
template <int d>
void readPointsFromBIN(FILE *file, int N, float **&points)
{
    printf("Wczytywanie punktów z pliku..\n");
    // Zapianie czasu początkowego wczytywania danych
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    size_t bytes_read;
    points = new float *[d];
    if (!points)
    {
        printf("Malloc error in function readPointsFromBIN with points\n");
        exit(1);
    }
    for (int i = 0; i < d; i++)
    {
        points[i] = new float[N];
        if (!points[i])
        {
            for (int j = 0; j < i; j++)
            {
                delete[] points[i];
            }
            delete[] points;
            printf("Malloc error in function readPointsFromBIN with points[%i]\n", i);
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < d; j++)
        {
            bytes_read = fread(&points[j][i], sizeof(float), 1, file);
            if (bytes_read != 1)
            {
                printf("Fread error in function readPointsFromBIN with points[%d][%d]\n.", j, i);
                exit(1);
            }
        }
    }
    fclose(file);

    // Zapisanie czasu końcowego wczytywania danych i różnicy między start a stop
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Czas wykonania: %f milisekund\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Wczytanie punktów w zależności od podanego formatu
template <int d>
void readPointsFromFile(const char *data_format, FILE *file, int N, float **&points)
{
    if (strcmp(data_format, "txt") == 0)
    {
        readPointsFromTXT<d>(file, N, points);
        return;
    }
    return readPointsFromBIN<d>(file, N, points);
}

// Zapisanie danych centroidów i przynależności punktów do centroidów do pliku dla danych przetworzonych przez algorytm wykonywany na CPU
template <int d>
void writeToFile(const char *filename_output, int N, int k, float **centroids, int *nearestCentroidID)
{
    printf("Zapisywanie danych do pliku...\n");
    // Zapisanie czasu początkowego
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *file = fopen(filename_output, "w");
    if (file == NULL)
    {
        printf("Unable to open file in function writeToFile\n");
        exit(1);
    }

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < d; j++)
        {
            fprintf(file, "%.4f ", centroids[j][i]);
        }
        fprintf(file, "\n");
    }
    for (int i = 0; i < N; i++)
    {
        fprintf(file, "%d\n", nearestCentroidID[i]);
    }

    fclose(file);

    // Zapisanie czasu końcowego i wypisanie w milisekundach
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Czas wykonania: %f milisekund\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Zapisanie danych centroidów i przynależności punktów do centroidów do pliku dla danych przetworzonych przez algorytm wykonywany na GPU
template <int d>
void writeToFile(const char *filename_output, int N, int k, float *centroids, int *nearestCentroidID)
{
    printf("Zapisywanie danych do pliku...\n");
    // Zapisanie czasu początkowego
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    FILE *file_output = fopen(filename_output, "w");
    if (file_output == NULL)
    {
        printf("Unable to open file in function writeToFile\n");
    }

    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < d; j++)
        {
            fprintf(file_output, "%.4f ", centroids[i + j * k]);
        }
        fprintf(file_output, "\n");
    }
    for (int i = 0; i < N; i++)
    {
        fprintf(file_output, "%d\n", nearestCentroidID[i]);
    }

    fclose(file_output);

    // Zapisanie czasu końcowego
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Czas wykonania: %f milisekund\n", time);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

