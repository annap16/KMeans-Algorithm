#include "file_manager.h"

void readHeaderFromTXT(FILE *file, int *NP, int *dP, int *kP)
{
    fscanf(file, "%d %d %d", NP, dP, kP);
}

void readHeaderFromBIN(FILE *file, int *NP, int *dP, int *kP)
{
    size_t bytes_read = fread(NP, sizeof(int), 1, file);
    if (bytes_read != 1)
    {
        printf("Fread error in function readHeaderFromBIN with NP\n.");
        exit(1);
    }

    bytes_read = fread(dP, sizeof(int), 1, file);
    if (bytes_read != 1)
    {
        printf("Fread error in function readHeaderFromBIN with dP\n.");
        exit(1);
    }

    bytes_read = fread(kP, sizeof(int), 1, file);
    if (bytes_read != 1)
    {
        printf("Fread error in function readHeaderFromBIN with kP\n.");
        exit(1);
    }
}