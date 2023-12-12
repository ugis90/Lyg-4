%%cu
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;

const int MAX_NAME_LENGTH = 100;
const int MAX_RESULT_STRING_LENGTH = 200;

struct Product {
    char Name[MAX_NAME_LENGTH];
    int Id;
    double Cost;
};

struct ProductResult {
    char Name[MAX_NAME_LENGTH];
    int Id;
    double Cost;
    char ComputedData[MAX_RESULT_STRING_LENGTH];
};

//Paleidzia ir vykdo GPU
__device__ void manualStrcpy(char* dest, const char* src) {
    int i = 0;
    while (src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0'; // Null-terminate the string
}

__device__ void floatToStr(double value, char* str) {
    // Basic conversion from a float to a string
    int intPart = static_cast<int>(value);
    double fracPart = value - static_cast<double>(intPart);
    int fracPartInt = static_cast<int>(fracPart * 1000000); // 6 decimal places

    // Convert integer part
    int i = 0;
    if (intPart == 0) {
        str[i++] = '0';
    } else {
        char temp[20];
        int j = 0;
        while (intPart != 0) {
            temp[j++] = '0' + (intPart % 10);
            intPart /= 10;
        }
        for (j = j - 1; j >= 0; j--) {
            str[i++] = temp[j];
        }
    }

    str[i++] = '.'; // Decimal point

    // Convert fractional part
    if (fracPartInt == 0) {
        str[i++] = '0';
    } else {
        char temp[20];
        int j = 0;
        while (fracPartInt != 0) {
            temp[j++] = '0' + (fracPartInt % 10);
            fracPartInt /= 10;
        }
        for (j = j - 1; j >= 0; j--) {
            str[i++] = temp[j];
        }
    }

    str[i] = '\0'; // Null-terminate the string
}

__global__ void CalculationsKernel(const Product* products, ProductResult* results, int* validCount, int numProducts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    int stride = blockDim.x * gridDim.x; // Total number of threads

    for (int i = idx; i < numProducts; i += stride) { // Loop through all products
        Product product = products[i];
        ProductResult result;
        double asciiSum = 0;

        // Calculate ASCII sum of the product name
        for (int j = 0; j < MAX_NAME_LENGTH && product.Name[j] != '\0'; ++j) {
            asciiSum += static_cast<int>(product.Name[j]);
        }

        // Calculate product.Cost to the power of 2
        double costPowerTwo = pow(product.Cost, 2);

        // Calculation combining ASCII sum and cost to the power of 2
        double combinedValue = asciiSum + costPowerTwo;

        // Check and add to results if combinedValue is above a certain threshold
        double threshold = 4000.0;
        if (combinedValue > threshold) {
            int insertIdx = atomicAdd(validCount, 1);

            // Copying the product data to result
            manualStrcpy(result.Name, product.Name);
            result.Id = product.Id;
            result.Cost = product.Cost;

            // Storing combinedValue as a string in ComputedData
            floatToStr(combinedValue, result.ComputedData);

            results[insertIdx] = result;
        }
    }
}

void RunCalculationsOnGPU(Product* products, ProductResult* results, int* validCount, int numProducts) {
    // Pointer to memory on the GPU
    Product* d_products;
    ProductResult* d_results;
    int* d_validCount;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_products, numProducts * sizeof(Product));
    cudaMalloc((void**)&d_results, numProducts * sizeof(ProductResult));
    cudaMalloc((void**)&d_validCount, sizeof(int));
    cudaMemset(d_validCount, 0, sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_products, products, numProducts * sizeof(Product), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int numBlocks = (numProducts + threadsPerBlock - 1) / threadsPerBlock;
    numBlocks = max(2, numBlocks); // At least two blocks

    // Run the kernel
    CalculationsKernel<<<numBlocks, threadsPerBlock>>>(d_products, d_results, d_validCount, numProducts);

    // Copy results from GPU to CPU
    cudaMemcpy(results, d_results, numProducts * sizeof(ProductResult), cudaMemcpyDeviceToHost);
    cudaMemcpy(validCount, d_validCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(d_products);
    cudaFree(d_results);
    cudaFree(d_validCount);
}

void ReadProducts(const string& dataFile, Product products[], int& index) {
    ifstream file(dataFile);

    if (!file.is_open()) {
        cout << "Failed to open the file: " << dataFile << endl;
        return;
    }

    index = 0;
    char semicolon;
    while (file >> ws && !file.eof()) {
        file.getline(products[index].Name, MAX_NAME_LENGTH, ';'); // Read until the semicolon
        file >> products[index].Id >> semicolon >> products[index].Cost; // Read the rest of the line
        file.ignore(numeric_limits<streamsize>::max(), '\n'); // Ignore the rest of the line
        index++;
    }

    file.close();
}

void PrintResults(const ProductResult results[], int validCount, const string& fileName) {
    ofstream out(fileName);

    out << "----------------------------------------------------------------------------------------" << endl;
    out << "| " << setw(33) << "Produktas" << " | " << setw(6) << "Id" << " | " << setw(6) << "Kaina" << " |" << setw(30) << "           Apskaičiuota reikšmė" << " |" << endl;
    out << "----------------------------------------------------------------------------------------" << endl;

    for (int i = 0; i < validCount; i++) {
        out << "| " << setw(33) << results[i].Name
            << " | " << setw(6) << results[i].Id
            << " | " << setw(6) << results[i].Cost
            << " | " << setw(30) << results[i].ComputedData << " |" << endl;
    }

    out << "----------------------------------------------------------------------------------------" << endl;
    out.close();
}

int main() {
    string inputFile1 = "./data/IFF-1-5_AndziulisJ_L1_dat_1.txt";
    string inputFile2 = "./data/IFF-1-5_AndziulisJ_L1_dat_2.txt";
    string inputFile3 = "./data/IFF-1-5_AndziulisJ_L1_dat_3.txt";
    string outputFile = "./output.txt"; // Replace with output file name
    int numProducts = 1000;
    Product* products = new Product[numProducts];
    ProductResult* results = new ProductResult[numProducts];
    int validCount = 0;

    ReadProducts(inputFile2, products, numProducts); // Replace with input file
    RunCalculationsOnGPU(products, results, &validCount, numProducts);
    PrintResults(results, validCount, outputFile);

    delete[] products;
    delete[] results;

    return 0;
}