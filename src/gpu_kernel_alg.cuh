#pragma once

template<int d>
void KMeansGPU1(float** points, int N, int k, const char* filename_output);

template<int d>
__global__ void kernelWithAtomicAdd(float* points, int* NP, int* kP, float* centroids, int* nearestCentroidID, int* centroidCounts, float* sumPerCentroid, int* changedP);

#ifdef IMPLEMENTATION

#include "file_manager.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>



// Algorytm k-średnich zaimplementowany na GPU za pomocą pamięci dzielonej i atomicAdd
template<int d>
void KMeansGPU1(float** points,int N, int k, const char* filename_output)
{
    cudaSetDevice(0);

    float* d_points=nullptr, *d_centroids=nullptr, *d_sumPerCentroid=nullptr, *sumPerCentroid=nullptr, *centroids=nullptr;
    int *d_nearestCentroidID=nullptr, *d_centroidCounts=nullptr, *centroidCounts=nullptr, *nearestCentroidID=nullptr;
    int *d_N=nullptr, *d_k=nullptr;

    // Block try wykorzystywany do wyłapania wyjątków pojawiających się podczas operacji zamkniętych w gpuErrchk
    try
    {
    // Alokacja pamięci na GPU
    gpuErrchk(cudaMalloc((void**)&d_points, sizeof(float)*N*d));
    gpuErrchk(cudaMalloc((void**)&d_centroids, sizeof(float)*k*d));
    gpuErrchk(cudaMalloc((void**)&d_sumPerCentroid, sizeof(float)*k*d));
    gpuErrchk(cudaMalloc((void**)&d_nearestCentroidID, sizeof(int)*N));
    gpuErrchk(cudaMalloc((void**)&d_centroidCounts, sizeof(int)*k));
    gpuErrchk(cudaMalloc((void**)&d_N, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_k, sizeof(int)));    

    for(int i=0; i<d; i++)
    {
        gpuErrchk(cudaMemcpy(d_points+i*N, points[i], sizeof(float)*N, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_centroids+i*k, points[i], sizeof(float)*k, cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaMemcpy(d_N, &N, sizeof(int),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_k, &k, sizeof(int),cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(d_sumPerCentroid, 0, sizeof(float)*k*d));
    gpuErrchk(cudaMemset(d_nearestCentroidID, -1, sizeof(int)*N));
    gpuErrchk(cudaMemset(d_centroidCounts, 0, sizeof(int)*k));

    // Obliczenie wielkości pamięci dzielonej i ilości potrzebnych bloków
    int smBytes = sizeof(float)*2*k*d + sizeof(int)*k;
    int threadsPerBlocks = 1024;
    int blocksCount = std::ceil(static_cast<float>(N)/1024.0);

    // Alokacja pamięci na CPU
    sumPerCentroid = new float[k*d];
    if(!sumPerCentroid)
    {
        printf("Malloc error in function KMeansGPU1 with sumPerCentroid\n");
        exit(1);
    }

    centroidCounts = new int[k];
    if(!centroidCounts)
    {
        printf("Malloc error in function KMeansGPU1 with centroidCounts\n");
        exit(1);
    }

    centroids = new float[d*k];
    if(!centroids)
    {
        printf("Malloc error in function KMeansGPU1 with centroids\n");
        exit(1);
    }

    bool anythingChanged = true;
    int counter=0;
    int changed=0;
    thrust::device_vector<int> d_changed_thrust(N);

    printf("Algorytm k-średnich na GPU - metoda opierająca się na pamięci dzielonej i operacji atomicAdd. Rozpoczęto obliczenia...\nIteracje:\n");
   
    // Zapisanie czasu początkowego
    cudaEvent_t start, stop; float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0);

    while(anythingChanged && counter<100)
    {
        counter++;         
        thrust::fill(d_changed_thrust.begin(),d_changed_thrust.end(), 0);

        // Wywołanie kernela odpowiadającego za znalezienie przynależności do klastra każdego z punktów oraz obliczanie sum i liczności w obrębie jednego centroidu
        kernelWithAtomicAdd<d><<<blocksCount, threadsPerBlocks, smBytes>>>(d_points, d_N, d_k, d_centroids, d_nearestCentroidID, d_centroidCounts, d_sumPerCentroid, thrust::raw_pointer_cast(d_changed_thrust.data()));
        
        // Sprawdzenie błędów i sychronizacja
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("CUDA Kernel failed: %s\n", cudaGetErrorString(err));
            exit(1); 
        }
        err = cudaDeviceSynchronize(); 
        if (err != cudaSuccess) 
        {
            printf("CUDA kernel execution failed: %s \n",cudaGetErrorString(err));
        }
        
        // Przekopiowanie sum współrzędnych punktów należących do klasta i liczności klastrów
        gpuErrchk(cudaMemcpy(sumPerCentroid, d_sumPerCentroid, sizeof(float)*k*d, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(centroidCounts, d_centroidCounts, sizeof(int)*k, cudaMemcpyDeviceToHost));  

        // Obliczenie współrzędnych nowych centroidów
        for(int i=0; i<k; i++)
        {
            for(int j=0; j<d; j++)
            {
                centroids[i+j*k] = sumPerCentroid[i+j*k]/(static_cast<float>(centroidCounts[i]));
            }
        }

        // Sprawdzenie ilości punktów które w tej iteracji zmieniły swoją przynależność
        changed = thrust::reduce(d_changed_thrust.begin(), d_changed_thrust.end(), 0, thrust::plus<int>());
        if(changed>0)
        {
            anythingChanged = true;
        }
        else
        {
            anythingChanged = false;
        }
        printf("Licznik: %d, Liczba pkt. która zmieniła przynależność: %d\n", counter, changed);
        if(!anythingChanged)
        {
            break;
        }

        // Przekopiowanie danych nowych centroidów na GPU i wyzerowanie tablic pomocniczych
        gpuErrchk(cudaMemset(d_sumPerCentroid, 0, sizeof(float)*k*d));
        gpuErrchk(cudaMemset(d_centroidCounts, 0, sizeof(int)*k));
        gpuErrchk(cudaMemcpy(d_centroids, centroids, sizeof(float)*k*d, cudaMemcpyHostToDevice));

    }

    // Zapisanie czasu końcowego
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("Czas wykonania: %f milisekund\n", time);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    nearestCentroidID = new int[N];
    if(!nearestCentroidID)
    {
        printf("Malloc error in function KMeansGPU1 with nearestCentroidID\n");
    }

    gpuErrchk(cudaMemcpy(nearestCentroidID, d_nearestCentroidID, sizeof(int)*N, cudaMemcpyDeviceToHost));

    // Zapisanie otrzymanych danych do pliku
    writeToFile<d>(filename_output, N, k, centroids, nearestCentroidID);

    // Dealokacja pamięci
    delete[] sumPerCentroid;
    delete[] centroids;
    delete[] centroidCounts;
    delete[] nearestCentroidID;
    for(int i=0; i<d; i++)
    {
        delete[] points[i];
    }
    delete[] points;


    cudaFree(d_N);
    cudaFree(d_k);
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_sumPerCentroid);
    cudaFree(d_nearestCentroidID);
    cudaFree(d_centroidCounts);
    }
    catch (const std::runtime_error& e) 
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        if(centroids)
        {
            delete[] centroids;
            centroids = nullptr;
        }
        if(nearestCentroidID)
        {
            delete[] nearestCentroidID;
            nearestCentroidID = nullptr;
        }
        if(sumPerCentroid)
        {
            delete[] sumPerCentroid;
            sumPerCentroid = nullptr;
        }
        if(centroidCounts)
        {
            delete[] centroidCounts;
            centroidCounts=nullptr;
        }
        for(int i=0; i<d; i++)
        {
            delete[] points[i];
            points[i]=nullptr;
        }
        delete[] points;
        points = nullptr;

        if(d_points)    cudaFree(d_points);
        if(d_centroids) cudaFree(d_centroids);
        if(d_nearestCentroidID) cudaFree(d_nearestCentroidID);
        if(d_sumPerCentroid) cudaFree(d_sumPerCentroid);
        if(d_centroidCounts) cudaFree(d_centroidCounts);
        if(d_N) cudaFree(d_N);
        if(d_k) cudaFree(d_k);
    }

}

// Kernel odpowiadający za znalezienie przynależności do klastra każdego z punktów oraz obliczanie sum i liczności w obrębie jednego centroidu
template<int d>
__global__ void kernelWithAtomicAdd(float* points, int* NP, int* kP, float* centroids, int* nearestCentroidID, int* centroidCounts, float* sumPerCentroid, int* changed)
{
    // Inicjalizacja zmiennych 
    int N = *NP;
    int k = *kP;
    int indx = blockIdx.x*1024 + threadIdx.x;
    int blockSize = blockDim.x;

    // Inicjalizacja pamięci dzielonej
    extern __shared__ float shmArray[];
    float* shmCentroids = (float*) shmArray;
    float* shmSumPerCentroid = (float*)&shmCentroids[k*d];
    int* shmCentroidCounts = (int*)&shmSumPerCentroid[k*d];
    
    // Wypełnienie pamięci dzielonej odpowiednimi wartościami
    for(int i=threadIdx.x; i<k*d;i+=blockSize)
    {
        shmCentroids[i] = centroids[i];
        shmSumPerCentroid[i] = 0;
    }
    for(int i=threadIdx.x; i<k; i+=blockSize)
    {
        shmCentroidCounts[i] = 0;
    }

    __syncthreads();

    if(indx>=N)
    {
        return;
    }

    float minDistance;
    int centroidID = 0;
    float sum = 0;
    // Obliczanie odległości punktu od zerowego centroidu
    for(int j = 0; j<d; j++)
    {
        sum += (shmCentroids[j*k]-points[indx+j*N])*(shmCentroids[j*k]-points[indx+j*N]);
    }
    minDistance = sum;

    // Znajdowanie centroidu znajdującego się najbliżej od danego punktu
    for(int i = 1; i<k; i++)
    {
        sum = 0;
        for(int j = 0; j<d; j++)
        {
            sum += (shmCentroids[i+j*k]-points[indx+j*N])*(shmCentroids[i+j*k]-points[indx+j*N]);
        }
        if(sum<minDistance)
        {
            minDistance = sum;
            centroidID = i;
        }
    }

    // Aktualizacja tablicy przynależności dla punktu i informacja o dokonanych zmianach
    if(nearestCentroidID[indx]!=centroidID)
    {
        changed[indx] = 1;
    }
    nearestCentroidID[indx] = centroidID;

    // Aktualizacja sumy dla każdej współrzędnej dla klastra, do którego należy dany punkt
    for(int i=0; i<d; i++)
    {
        atomicAdd(&shmSumPerCentroid[centroidID+i*k], points[indx+i*N]);
    }

    // Aktualizacja liczności klastra
    atomicAdd(&shmCentroidCounts[centroidID], 1);
    
    __syncthreads();

    // Aktualizacja sum i liczności klastrów w pamięci globalnej
    for(int i=threadIdx.x; i<k*d;i+=blockSize)
    {
        atomicAdd(&sumPerCentroid[i], shmSumPerCentroid[i]);
    }
    for(int i=threadIdx.x; i<k; i+=blockSize)
    {
        atomicAdd(&centroidCounts[i], shmCentroidCounts[i]);
    }
}

#endif