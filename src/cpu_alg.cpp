// #include <stdio.h>
// #include <stdlib.h>
// #include <cuda_runtime.h>
// #include "cpu_alg.h"


// // Algorytm k-średnich zaimplementowany na CPU
// template<int d>
// void KMeansCPU(float** points, int N, int k, const char* filename_output)
// {
//     // Zaalokowanie potrzebnych tablic
//     float** centroids = new float*[d];
//     if(!centroids)
//     {
//         printf("Malloc error in function KMeansCPU with centroids\n");
//         for(int i=0; i<d; i++)
//         {
//             delete[] points[i];
//         }
//         delete[] points;
//         exit(1);
//     }
//     for(int i=0; i<d;i++)
//     {
//         centroids[i] =  new float[k];
//         if(!centroids[i])
//         {
//             for(int j=0; j<i; i++)
//             {
//                 delete[] centroids[i];
//             }
//             delete[] centroids;
//             for(int j=0; j<d; j++)
//             {
//                 delete[] points[j];
//             }
//             delete[] points;
//             printf("Malloc error in function KMeansCPU with centroids[%d]\n", i);
//             exit(1);
//         }
//     }

//     int *nearestCentroidID = new int[N];
//     if(!nearestCentroidID)
//     {
//         printf("Malloc error in function KMeansCPU with nearestCentroidID\n");
//         for(int i=0; i<d; i++)
//         {
//             delete[] points[i];
//             delete[] centroids[i];
//         }
//         delete[] points;
//         delete[] centroids;
//         exit(1);
//     }
//     float** sumPerCentroid = new float*[d];
//     if(!sumPerCentroid)
//     {
//         printf("Malloc error in function KMeansCPU with nearestCentroidID\n");
//         for(int i=0; i<d; i++)
//         {
//             delete[] points[i];
//             delete[] centroids[i];
//         }
//         delete[] points;
//         delete[] centroids;
//         delete[] nearestCentroidID;
//         exit(1);
//     }

//     for(int i=0; i<d; i++)
//     {
//         sumPerCentroid[i] = new float[k];
//         if(!sumPerCentroid[i])
//         {
//             for(int j=0; j<i; j++)
//             {
//                 delete[] sumPerCentroid[i];
//             }
//             for(int j=0; j<d; j++)
//             {
//                 delete[] points[j];
//                 delete[] centroids[j];
//             }
//             delete[] sumPerCentroid;
//             delete[] points;
//             delete[] centroids;
//             delete[] nearestCentroidID;
//             printf("Malloc error in function KMeansCPU with nearestCentroidID[%d]\n", i);
//         }
//     }
//     int* centroidCounts = new int[k];
//     if(!centroidCounts)
//     {
//         printf("Malloc error in function KMeansCPU with centroidCounts\n");
//         for(int j=0; j<d; j++)
//         {
//             delete[] points[j];
//             delete[] centroids[j];
//             delete[] sumPerCentroid[j];
//         }
//         delete[] sumPerCentroid;
//         delete[] points;
//         delete[] centroids;
//         delete[] nearestCentroidID;
//         exit(1);   
//     }

//     // Inicjalizowanie tablic i zmiennych odpowiednimi wartościami
//     for(int i=0; i<N;i++)
//     {
//         nearestCentroidID[i] = -1;
//     }
//     for(int i=0; i<k; i++)
//     {
//         for(int j=0; j<d; j++)
//         {
//             centroids[j][i] = points[j][i];
//         }
//     }
//     int changed;
//     bool anythingChanged = true;
//     int counter=0;

//     printf("Algorytm k-średnich na CPU. Rozpoczęto obliczenia...\nIteracje:\n");
//     // Zapisanie czasu początkowego
//     cudaEvent_t start, stop; float time;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord( start, 0);


//     // Główna pętla programu kontrolująca zmiany przynależności punktów do klastrów, implementacja algorytmu k-średnich
//     while(anythingChanged && counter<100)
//     {
//         counter++;
        
//         // Wyzerowanie tablic potrzebnych do obliczania sum w obrębie centroidu oraz liczności centroidu
//         for(int i=0; i<k;i++)
//         {
//             for(int j=0; j<d; j++)
//             {
//                 sumPerCentroid[j][i] = 0;
//             }
//         }
//         for(int i=0; i<k;i++)
//         {
//             centroidCounts[i] = 0;
//         }

//         anythingChanged = false;
//         changed=0;
//         // Obliczanie przynależności każdego punktu do centroidu
//         for(int i=0; i<N;i++)
//         {
//             int newCentroidID = findClosestCentroidID<d>(i, points, centroids, N, k);
//             if(newCentroidID!=nearestCentroidID[i])
//             {
//                 anythingChanged = true;
//                 changed++;
//             }
//             nearestCentroidID[i] = newCentroidID;
//             centroidCounts[newCentroidID]++;

//             // Obliczanie sumy dla centroidu, do którego należy analizowany punkt
//             for(int j=0; j<d; j++)
//             {
//                 sumPerCentroid[j][newCentroidID] += points[j][i];
//             }
//         }
//         printf("Licznik:%d, Liczba pkt. która zmieniła przynależność: %d\n", counter, changed);

//         // Obliczanie nowych centroidów
//         for(int i=0; i<k; i++)
//         {
//             for(int j=0; j<d; j++)
//             {
//                 centroids[j][i] = sumPerCentroid[j][i]/centroidCounts[i];
//             }
//         }
//     }
//     // Zapisanie czasu końcowego
//     cudaEventRecord( stop, 0 );
//     cudaEventSynchronize( stop );
//     cudaEventElapsedTime( &time, start, stop );
//     printf("Czas wykonania: %f milisekund\n", time);
//     cudaEventDestroy( start );
//     cudaEventDestroy( stop );

//     // Zapisanie otrzymanych wyników do pliku
//     writeToFile<d>(filename_output, N, k , centroids, nearestCentroidID);

//     // Dealokacja pamięci
//     for(int i=0; i<d; i++)
//     {
//         delete[] sumPerCentroid[i];
//         delete[] centroids[i];
//     }
//     delete[] sumPerCentroid;
//     delete[] centroids;
//     delete[] nearestCentroidID;
//     delete[] centroidCounts;
//     for(int i=0; i<d; i++)
//     {
//         delete[] points[i];
//     }
//     delete[] points;
// }

// // Funkcja znajdująca najbliżej położony centroid względem podanego punktu
// template<int d>
// int findClosestCentroidID(int indx, float** points, float** centroids, int N, int k)
// {
//     double minDistance;
//     int centroidID = 0;
//     double sum = 0;

//     // Obliczenie odległości od centroidu zerowego i inicjalizacja minDistance
//     for(int j = 0; j<d; j++)
//     {
//         sum += (centroids[j][0]-points[j][indx])*(centroids[j][0]-points[j][indx]);
//     }
//     minDistance = sum;

//     // Znalezienie centroidu położonego najbliżej punktu
//     for(int i = 1; i<k; i++)
//     {
//         sum = 0;
//         for(int j = 0; j<d; j++)
//         {
//             sum += (centroids[j][i]-points[j][indx])*(centroids[j][i]-points[j][indx]);
//         }
//         if(sum<minDistance)
//         {
//             minDistance = sum;
//             centroidID = i;
//         }
//     }
//     return centroidID;
// }





