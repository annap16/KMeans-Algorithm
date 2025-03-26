#pragma once


template<int d>
void KMeansGPU2(float** points, int N, int k, const char* filename_output);

template<int d>
__global__ void kernelNearestCentroid(float* points, int* NP, int* kP, float* centroids, int* d_nearestCentroidID, int* changedP);


#ifdef IMPLEMENTATION

#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include "file_manager.h"


// Algorytm k-średnich zaimplementowany na GPU za pomocą biblioteki thrust
template<int d>
void KMeansGPU2(float** points,int N, int k, const char* filename_output)
{
    cudaSetDevice(0);  
    float* d_points=nullptr, *d_centroids=nullptr, *centroids=nullptr;
    int *d_nearestCentroidID=nullptr, *nearestCentroidID=nullptr;
    int *d_N=nullptr, *d_k=nullptr;

    // Block try wykorzystywany do wyłapania wyjątków pojawiających się podczas operacji zamkniętych w gpuErrchk
    try
    {

    // Alokacja pamięci na GPU
    gpuErrchk(cudaMalloc((void**)&d_points, sizeof(float)*N*d));
    gpuErrchk(cudaMalloc((void**)&d_centroids, sizeof(float)*k*d));
    gpuErrchk(cudaMalloc((void**)&d_nearestCentroidID, sizeof(int)*N));
    gpuErrchk(cudaMalloc((void**)&d_N, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_k, sizeof(int)));

    for(int i=0; i<d; i++)
    {
        gpuErrchk(cudaMemcpy(d_points+i*N, points[i], sizeof(float)*N, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_centroids+i*k, points[i], sizeof(float)*k, cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaMemcpy(d_N, &N, sizeof(int),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_k, &k, sizeof(int),cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_nearestCentroidID, -1, sizeof(int)*N));

    
    // Obliczenie wielkości pamięci dzielonej i ilości potrzebnych bloków
    int smBytes = sizeof(float)*k*d;
    int threadsPerBlocks = 1024;
    int blocksCount = std::ceil(static_cast<float>(N)/1024.0);

    // Alokacja pamięci na CPU
    centroids = new float[d*k];
    if(!centroids)
    {
        printf("Malloc error in function KMeansGPU2 with centroids\n");
        exit(1);
    }
    nearestCentroidID = new int[N];
    if(!nearestCentroidID)
    {
        printf("Malloc error in function KMeansGPU2 with nearestCentroidID\n");
        exit(1);
    }

    bool anythingChanged = true;
    int counter=0;
    int changed=0;

    // Tworzenie wektorów z biblioteki thrust oraz ich inicjalizacja
    thrust::device_vector<int> d_changed_thrust(N);
    thrust::device_vector<float> d_pointsCopy(N*d);
    thrust::device_vector<int> d_nearestCentroidIDCopy(N);
    thrust::device_vector<float> d_centroidSums(N);
    thrust::device_vector<int> d_keysAferReduce(N);
    thrust::device_vector<int> d_centroidCountVec(N);
    thrust::device_vector<float>d_onesVec(N);
    thrust::fill(d_onesVec.begin(), d_onesVec.end(), 1.0);
    thrust::copy(d_points, d_points+d*N, d_pointsCopy.begin());
    thrust::copy(d_nearestCentroidID, d_nearestCentroidID+N, d_nearestCentroidIDCopy.begin());

    printf("Algorytm k-średnich na GPU - metoda opierająca się na bibliotece thrust. Rozpoczęto obliczenia...\nIteracje:\n");
    // Zapisanie czasu początkowego
    cudaEvent_t start, stop; float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0);

    while(anythingChanged && counter<100)
    {
        counter++;

        thrust::fill(d_changed_thrust.begin(),d_changed_thrust.end(), 0);

        // Wywołanie kernela odpowiadającego za znalezienie przynależności do klastra każdego z punktów
        kernelNearestCentroid<d><<<blocksCount, threadsPerBlocks, smBytes>>>(d_points, d_N, d_k, d_centroids, d_nearestCentroidID, thrust::raw_pointer_cast(d_changed_thrust.data()));
        
        // Sprawdzenie błędów i sychronizacja
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Kernel failed: %s\n", cudaGetErrorString(err));
            exit(1);  
        }
        err = cudaDeviceSynchronize(); 
        if (err != cudaSuccess) 
        {
            printf("CUDA kernel execution failed: %s \n",cudaGetErrorString(err));
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

        // Skopiowanie danych punktów i przynależności punktów do centroidów przed sortowaniem, aby móc później odzyskać ich pierwotną kolejność
        thrust::copy(thrust::device, d_points, d_points+d*N, d_pointsCopy.begin());
        thrust::copy(thrust::device, d_nearestCentroidID, d_nearestCentroidID+N, d_nearestCentroidIDCopy.begin());

        // Obliczanie ilości punktów w każdym klastrze
        thrust::sort(thrust::device, d_nearestCentroidIDCopy.begin(), d_nearestCentroidIDCopy.end(), thrust::less<int>());
        thrust::reduce_by_key(thrust::device , d_nearestCentroidIDCopy.begin(), d_nearestCentroidIDCopy.end(), d_onesVec.begin(), d_keysAferReduce.begin(), d_centroidCountVec.begin(), thrust::equal_to<int>(),   
            thrust::plus<int>());
        
        for(int i=0; i<d; i++)
        {
            
            // Skopiowanie przynależności punktów do centroidów przed sortowaniem, aby móc później odzyskać ich pierwotną kolejność
            thrust::copy(thrust::device, d_nearestCentroidID, d_nearestCentroidID+N, d_nearestCentroidIDCopy.begin());

            // Obliczanie sum dla odpowiedniej współrzędnej każdego centroidu, aby później wyliczyć średnią i nowe centroidy
            thrust::stable_sort_by_key(thrust::device, d_nearestCentroidIDCopy.begin(), d_nearestCentroidIDCopy.end(),  d_pointsCopy.begin()+i*N, thrust::less<int>());
            thrust::reduce_by_key(thrust::device , d_nearestCentroidIDCopy.begin(), d_nearestCentroidIDCopy.end(), d_pointsCopy.begin()+i*N, d_keysAferReduce.begin(), d_centroidSums.begin(), thrust::equal_to<int>(), 
                thrust::plus<float>()    );
            

            // Jeżeli do każdego centroidu należy conajmniej jeden punkt średnia obliczana jest równocześnie dla wszystkich danych, a wyniki są zapisywane
            // od razu do tablicy, ponieważ indeksacja nie została zaburzona. W przeciwnym przypadku średnie są obliczane oddzielnie i zapisywane do konkretnych 
            // komórek w tablicy
            if(d_centroidCountVec[k-1]!=0)
            {
                thrust::transform(thrust::device, d_centroidSums.begin(), d_centroidSums.begin()+k , d_centroidCountVec.begin(), d_centroids+i*k, thrust::divides<float>());
            }
            else
            {
                for(int j=0; j<k;j++)
                {
                    if(d_centroidCountVec[j]!=0)
                    {
                        float newCentroidTemp = d_centroidSums[j]/d_centroidCountVec[j];
                        gpuErrchk(cudaMemcpy(d_centroids + i*k+d_keysAferReduce[j], &newCentroidTemp, sizeof(float), cudaMemcpyHostToDevice));
                    }
                }
            }
        }

    }
    // Zapisanie czasu końcowego
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop );
    printf("Czas wykonania: %f milisekund\n", time);
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    
    // Kopiowanie danych na CPU
    gpuErrchk(cudaMemcpy(nearestCentroidID, d_nearestCentroidID, sizeof(int)*N, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(centroids, d_centroids, sizeof(float)*k*d, cudaMemcpyDeviceToHost));

    // Zapisanie otrzymanych danych do pliku
    writeToFile<d>(filename_output, N, k, centroids, nearestCentroidID);

    // Dealokacja pamięci
    delete[] centroids;
    delete[] nearestCentroidID;
    for(int i=0; i<d; i++)
    {
        delete[] points[i];
    }
    delete[] points;

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_nearestCentroidID);
    cudaFree(d_N);
    cudaFree(d_k);
    }
    catch(const std::runtime_error& e)
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
        if(d_N) cudaFree(d_N);
        if(d_k) cudaFree(d_k);
    }


}


// Kernel odpowiadający za znalezienie przynależności do klastra każdego z punktów
template<int d>
__global__ void kernelNearestCentroid(float* points, int* NP, int* kP, float* centroids, int* d_nearestCentroidID, int* changedP)
{   
    // Inicjalizacja zmiennych 
    int N = *NP;
    int k = *kP;
    int indx = blockIdx.x*1024 + threadIdx.x;
    int blockSize = blockDim.x;

    // Inicjalizacja pamięci dzielonej
    extern __shared__ float shmArray[];
    float* shmCentroids = (float*) shmArray;
    
    // Wypełnienie pamięci dzielonej odpowiednimi wartościami
    for(int i=threadIdx.x; i<k*d;i+=blockSize)
    {
        shmCentroids[i] = centroids[i];
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

    // Aktualizacja tablicy przynależności dla punktu i informacji o dokonanych zmianach
    if(d_nearestCentroidID[indx]!=centroidID)
    {
        changedP[indx] = 1;
    }
    d_nearestCentroidID[indx] = centroidID;

}




#endif