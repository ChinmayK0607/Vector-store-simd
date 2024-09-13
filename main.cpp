#include <bits/stdc++.h>
#include <immintrin.h>
#include <chrono>
#include <thread>
using namespace std;

class Vectorstore {
private:
    unordered_map<int, vector<float>> vector_data;
    unordered_map<int, unordered_map<int, float>> vector_index;

    float horizontalAdd(__m256 vec) const {
        __m128 hi = _mm256_extractf128_ps(vec, 1);
        __m128 lo = _mm256_castps256_ps128(vec);
        lo = _mm_add_ps(lo, hi);

        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);

        return _mm_cvtss_f32(sums);
    }

    float computeNorm(const vector<float>& vec) const {
        size_t size = vec.size();
        __m256 sum = _mm256_setzero_ps();  

        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 v = _mm256_loadu_ps(&vec[i]);
            sum = _mm256_fmadd_ps(v, v, sum);
        }

        float totalSum = horizontalAdd(sum);

        for (; i < size; ++i) {
            totalSum += vec[i] * vec[i];
        }

        return sqrt(totalSum);
    }

    float dotProduct(const vector<float>& A, const vector<float>& B) const {
        if (A.size() != B.size()) {
            throw invalid_argument("Vectors must have the same size");
        }

        size_t size = A.size();
        __m256 sum = _mm256_setzero_ps();  

        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 vecA = _mm256_loadu_ps(&A[i]);
            __m256 vecB = _mm256_loadu_ps(&B[i]);
            sum = _mm256_fmadd_ps(vecA, vecB, sum);
        }

        float totalSum = horizontalAdd(sum);

        for (; i < size; ++i) {
            totalSum += A[i] * B[i];
        }

        return totalSum;
    }

public:
    Vectorstore() {}

    float Cosimilar(const vector<float>& A, const vector<float>& B) const {
        if (A.size() != B.size()) {
            throw invalid_argument("Vector A and B must have the same size");
        }

        size_t size = A.size();
        __m256 sum_dot = _mm256_setzero_ps();
        __m256 sum_normA = _mm256_setzero_ps();
        __m256 sum_normB = _mm256_setzero_ps();

        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 vecA = _mm256_loadu_ps(&A[i]);
            __m256 vecB = _mm256_loadu_ps(&B[i]);

            sum_dot = _mm256_fmadd_ps(vecA, vecB, sum_dot);
            sum_normA = _mm256_fmadd_ps(vecA, vecA, sum_normA);
            sum_normB = _mm256_fmadd_ps(vecB, vecB, sum_normB);
        }

        float dot = horizontalAdd(sum_dot);
        float normA = horizontalAdd(sum_normA);
        float normB = horizontalAdd(sum_normB);

        for (; i < size; ++i) {
            dot += A[i] * B[i];
            normA += A[i] * A[i];
            normB += B[i] * B[i];
        }

        normA = sqrt(normA);
        normB = sqrt(normB);

        if (normA == 0.0f || normB == 0.0f) {
            return 0.0f;
        }

        return dot / (normA * normB);
    }

    vector<float> getVector(int id) {
        return vector_data[id];
    }

    void setVector(int id, const vector<float>& vec) {
        vector_data[id] = vec;
        updateVector(vec, id);
    }

    void updateVector(const vector<float>& vector, int id) {
        for (const auto& [existing_id, existing_vector] : vector_data) {
            if (existing_id != id) {
                float similarity = Cosimilar(vector, existing_vector);
                vector_index[id][existing_id] = similarity;
                vector_index[existing_id][id] = similarity;
            }
        }
        vector_data[id] = vector;
    }

    vector<pair<int, float>> findSimilar(const vector<float>& query, int num_results) {
        vector<pair<int, float>> scores;
        for (const auto& [id, vec] : vector_data) {
            float score = dotProduct(vec, query);
            score = score / (computeNorm(query) * computeNorm(vec));
            scores.push_back({id, score});
        }

        partial_sort(scores.begin(), scores.begin() + min(num_results, static_cast<int>(scores.size())), scores.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
        scores.resize(min(num_results, static_cast<int>(scores.size())));
        return scores;
    }

    // Batch processing function
    vector<vector<float>> batchCosimilar(const vector<vector<float>>& batchA, const vector<vector<float>>& batchB) const {
        size_t sizeA = batchA.size();
        size_t sizeB = batchB.size();
        vector<vector<float>> result(sizeA, vector<float>(sizeB));

        // Multithreading
        unsigned int numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4;

        vector<thread> threads;
        size_t chunkSize = sizeA / numThreads;
        size_t remainder = sizeA % numThreads;

        size_t startIndex = 0;

        for (unsigned int t = 0; t < numThreads; ++t) {
            size_t endIndex = startIndex + chunkSize + (t < remainder ? 1 : 0);
            threads.emplace_back([this, &batchA, &batchB, &result, startIndex, endIndex]() {
                for (size_t i = startIndex; i < endIndex; ++i) {
                    for (size_t j = 0; j < batchB.size(); ++j) {
                        result[i][j] = Cosimilar(batchA[i], batchB[j]);
                    }
                }
            });
            startIndex = endIndex;
        }

        // Join threads
        for (auto& th : threads) {
            th.join();
        }

        return result;
    }
};

void testVectorstorePerformance(int numVectors, int vectorSize) {
    Vectorstore store;

    vector<vector<float>> vectors;
    srand(time(0));
    for (int i = 1; i <= numVectors; ++i) {
        vector<float> vec(vectorSize);
        generate(vec.begin(), vec.end(), []() { return static_cast<float>(rand()) / RAND_MAX; });
        store.setVector(i, vec);
        vectors.push_back(vec);
    }

    auto start = chrono::high_resolution_clock::now();

    // Multithreading implementation
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4; // Fallback to 4 threads if hardware_concurrency can't detect

    vector<thread> threads;
    size_t chunkSize = numVectors / numThreads;
    size_t remainder = numVectors % numThreads;

    size_t startIndex = 0;

    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t endIndex = startIndex + chunkSize + (t < remainder ? 1 : 0);
        threads.emplace_back([&store, &vectors, startIndex, endIndex, numVectors]() {
            for (size_t i = startIndex; i < endIndex; ++i) {
                for (size_t j = i + 1; j < numVectors; ++j) {
                    float similarity = store.Cosimilar(vectors[i], vectors[j]);
                    (void)similarity; // Suppress unused variable warning
                }
            }
        });
        startIndex = endIndex;
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Time taken to compute cosine similarities for " << numVectors << " vectors using multithreading: "
         << elapsed.count() << " seconds." << endl;
}

void testBatchProcessing(int numVectors, int vectorSize, int batchSize) {
    Vectorstore store;

    vector<vector<float>> vectors;
    srand(time(0));
    for (int i = 1; i <= numVectors; ++i) {
        vector<float> vec(vectorSize);
        generate(vec.begin(), vec.end(), []() { return static_cast<float>(rand()) / RAND_MAX; });
        store.setVector(i, vec);
        vectors.push_back(vec);
    }

    // Generate a batch of query vectors
    vector<vector<float>> queryBatch(batchSize);
    for (int i = 0; i < batchSize; ++i) {
        vector<float> vec(vectorSize);
        generate(vec.begin(), vec.end(), []() { return static_cast<float>(rand()) / RAND_MAX; });
        queryBatch[i] = vec;
    }

    auto start = chrono::high_resolution_clock::now();

    // Compute batch similarities
    vector<vector<float>> similarities = store.batchCosimilar(queryBatch, vectors);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Time taken to compute batch cosine similarities for " << batchSize << " queries against "
         << numVectors << " vectors: " << elapsed.count() << " seconds." << endl;
}

int main() {
    int numVectors = 10000;
    int vectorSize = 128;
    int batchSize = 1000;

    cout << "Starting performance test with multithreading..." << endl;
    testVectorstorePerformance(numVectors, vectorSize);
    cout << "Performance test completed." << endl;

    cout << "\nStarting batch processing test..." << endl;
    testBatchProcessing(numVectors, vectorSize, batchSize);
    cout << "Batch processing test completed." << endl;

    return 0;
}
// Compile with: g++ -std=c++17 -O3 -mavx2 -pthread test1.cpp -o vectorstore
// g++ -std=c++17 -O2 -mavx -mfma -msse -msse2 -o test.exe test1.cpp
