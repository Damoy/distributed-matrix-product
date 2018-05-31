/*

		----------------------------------------------------

			-- Distributed matrix project --
			
			Author: FORNALI Damien
			Grade: Master I - IFI
			University: Nice-Sophia-Antipolis
			Year: 2017-2018
			Project subject link: https://sites.google.com/site/fabricehuet/teaching/parallelisme-et-distribution/projet---produit-matriciel-distribue

		----------------------------------------------------

*/

// mpicc -fopenmp -Wall mdot.c ./mdot -lm
// mpirun -np 4 --oversubscribe ./mdot

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>


/* -- Structures -- */

struct matrix{
	long* data;
	unsigned long numRows;
	unsigned long numCols;
};

struct vector{
	long* data;
	unsigned long dimension;
};

/* -- Declarations -- */

// allocation and destruction
struct matrix* allocateMatrix(unsigned long numRows, unsigned long numCols);
struct vector* allocateVector(unsigned long dimension);
struct vector* createVector(unsigned long dimension, long* data);
void destroyMatrix(struct matrix* mat);
void destroyVector(struct vector* vec);

// extraction
struct vector* extractColumn(struct matrix* mat, unsigned long col);
struct vector* extractRow(struct matrix* mat, unsigned long row);
long* lightExtractRow(struct matrix* mat, unsigned long row);

// calculations
void mdot(int rank, int numProcs, const char* Apath, const char* Bpath);
long vdot(unsigned long dimension, long* A_data, long* B_data);

// 'A' matrix rows permutations methods
// void p0_permutation1(unsigned long maxProcs, unsigned long dimension, unsigned long procNumData, unsigned long nextProcNumData,
//         unsigned long prevProcNumData, struct matrix* arows, struct matrix* bcols, struct matrix* result);
// void p0_permutation2(unsigned long maxProcs, unsigned long dimension, unsigned long procNumData, unsigned long* procNumDataBuffer,
//         struct matrix* arows, struct matrix* bcols, struct matrix* result);
void p0_classic_permutation(unsigned long maxProcs, unsigned long dimension, unsigned long procNumData, unsigned long* procNumDataBuffer,
        struct matrix* arows, struct matrix* bcols, struct matrix* result);
void pX_classic_permutation(int rank, unsigned long maxProcs, unsigned long dimension, unsigned long procNumData,
		struct matrix* arows, struct matrix* bcols, struct matrix* result);

// normalized gets and sets
long nget(struct matrix* mat, long row, long col);
void nset(struct matrix* mat, long row, long col, long value);
void nsetVec(struct matrix* mat, long row, struct vector* data);
void nsetVecLight(struct matrix* mat, long row, long* data, unsigned long size);
long* ngetVec(struct matrix* mat, long row);
void nsetData(struct matrix* mat, long row, long* data, unsigned long length);

// mpi wrappers
int mpiSend(const void* buf, int count, int dest);
int mpiRecv(void* buf, int count, int source);

// parsing
struct matrix* parse(const char* filePath);

// prints
void printMatrix(struct matrix* mat);


// -------------------- Main --------------------

int main(int argc, char** argv){
	if(argc != 3){
		printf("Wrong number of arguments.\n");
		return 1;
	}

	int numProcs;
	int rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	mdot(rank, numProcs, argv[1], argv[2]);

	MPI_Finalize();
	return 0;
}

// ----------------------------------------------


/* -- Calculations -- */

/**
	The matrices multiplication.

	rank: the processor rank
	numProcs: the total count of processors given by the user
	Apath: the first matrix path to use in calculation
	Bparh: the second one
*/
void mdot(int rank, int numProcs, const char* Apath, const char* Bpath) {
	struct matrix* A;
	struct matrix* B;

	// matrices dimension
	unsigned long dimension;
	// number of rows / columns the processor will handle
	unsigned long procNumData;

	// maximum of processors used through the code
	unsigned long maxProcs;

	// reference for the rank 0
	// number of data (rows / columns) of each processor
	unsigned long* procNumDataBuffer;

	if(rank == 0){
		// only processor 0 parses the matrices
		A = parse(Apath);
		B = parse(Bpath);

		dimension = B->numCols;

		/* computes the number of rows and columns each processor will process */

		if(dimension % numProcs == 0){
			// N multiple of P

			// first case, dimension is a multiple of numProcs
			procNumData = dimension / numProcs;
			// and the processors used is the number inquired in the mpirun command
			maxProcs = numProcs;

			// each processor needs to know if it can work or exit
			for(unsigned long p = 1; p < numProcs; ++p) // was maxProcs
				mpiSend(&maxProcs, 1, p);

			for(unsigned long p = 1; p < numProcs; ++p)
				mpiSend(&procNumData, 1, p);

        	procNumDataBuffer = malloc(sizeof(unsigned long) * maxProcs);

			// fill the buffer containing all the number of data information
			#pragma omp parallel for
			for(unsigned long i = 0; i < maxProcs; ++i)
				procNumDataBuffer[i] = procNumData;

		} else if(numProcs > dimension){
			// P > N

			// second case, processors are bigger than matrices dimension
			// ignores the numProcs..(numProcs + (numProcs - dimension) - 1)) procs
			procNumData = 1;
			// the dimension limits the number of processors, others are ignored
			maxProcs = dimension;

			// each processor needs to know if it can work or exit
			for(unsigned long p = 1; p < numProcs; ++p)
				mpiSend(&maxProcs, 1, p);

			// only working ones need to know their processing amount of data
			for(unsigned long p = 1; p < maxProcs; ++p)
				mpiSend(&procNumData, 1, p);

			procNumDataBuffer = malloc(sizeof(unsigned long) * maxProcs);

			// fill the buffer containing all the number of data information
			#pragma omp parallel for
			for(unsigned long i = 0; i < maxProcs; ++i)
				procNumDataBuffer[i] = procNumData;

		} else {
			// N > P && N mod P != 0

			// third case, dimension > numProcs && dimension % numProcs != 0
			maxProcs = numProcs;

			// each processor needs to know if it can work or exit
			for(unsigned long p = 1; p < numProcs; ++p)
				mpiSend(&maxProcs, 1, p);

			// all processors will have at least baseSize as number of data (number of A-rows and B-columns) to process
			unsigned long baseSize = dimension / numProcs;
			// used to know if a processor has to handle more than others
			unsigned long maxProcUpBound = dimension % numProcs;

			procNumDataBuffer = malloc(sizeof(unsigned long) * maxProcs);

			// the number of rows / cols handled by the processor 0
			procNumData = baseSize + 1;
			procNumDataBuffer[0] = procNumData;

			// used to send to a processor its number of data
			unsigned long numDataBuffer;

			for(unsigned long p = 1; p < numProcs; ++p){
				numDataBuffer = baseSize;

				// each processor which id is inferior to the max bound
				// will handle one more row and column than others
				if(p < maxProcUpBound)
					++numDataBuffer;

				mpiSend(&numDataBuffer, 1, p);

				// fill the buffer containing all the number of data information
				procNumDataBuffer[p] = numDataBuffer;
			}
		}

		// the others processors need the dimension information
		for(unsigned long p = 1; p < numProcs; ++p){
			mpiSend(&dimension, 1, p);
		}
	}
	 
	// first, all processors need to know the number of processors that will compute
	if(rank != 0){
		mpiRecv(&maxProcs, 1, 0);
	}
	
	if(rank > 0 && rank < maxProcs){
		// other processors receive the dimension and
		// number of rows / columns it will handle
		mpiRecv(&procNumData, 1, 0);
		mpiRecv(&dimension, 1, 0);
	} else if(rank >= maxProcs){
		// "terminate" the non-working ones
		return;
	}

	// used to store information in order to be able to send it
    // to other processors
    struct matrix* arows = allocateMatrix(procNumData, dimension);
    struct matrix* bcols = allocateMatrix(procNumData, dimension);
	// the processor calculations result
	struct matrix* result = allocateMatrix(procNumData, dimension);

	// -- first processor --
    if(rank == 0){

		// initial processor 0 data extraction
        #pragma omp parallel for
		for(unsigned long i = 0; i < procNumData; ++i){
			// get the initial proc data
			long* row = lightExtractRow(A, i);
            struct vector* col = extractColumn(B, i);

            nsetVecLight(arows, i, row, dimension);
            nsetVec(bcols, i, col);

            destroyVector(col);
		}


		// create the result matrix
        struct matrix* C = allocateMatrix(dimension, dimension);

        // initial distribution of data
		for(unsigned long p = 1; p < maxProcs; ++p){
			for(unsigned long idata = 0; idata < procNumDataBuffer[p]; ++idata){
				// extract a processor rows and cols and send them to it
				long* procRowData = lightExtractRow(A, (p * procNumDataBuffer[p]) + idata);
				mpiSend(procRowData, dimension, p);

				struct vector* procCol = extractColumn(B, (p * procNumDataBuffer[p]) + idata);
				mpiSend(procCol->data, dimension, p);

                destroyVector(procCol);
			}
        }

		
		// Set the initial matrix dot results
        #pragma omp parallel for
		for(unsigned long i = 0; i < procNumData; ++i)
            for(unsigned long j = 0; j < procNumData; ++j)
                nset(result, i, j, vdot(dimension, ngetVec(arows, j), ngetVec(bcols, i)));

        // [3]. first processor previous and next number of data
        // unsigned long prevProcNumData = procNumDataBuffer[maxProcs - 1];
        // unsigned long nextProcNumData = procNumDataBuffer[1];

		// [3]. the others processors need to know (part 3) the number of data their previous
		// and next ones handle
		for(unsigned long p = 1; p < maxProcs; ++p){
			unsigned long pPrevProcNumData = procNumDataBuffer[(p - 1 + maxProcs) % maxProcs];
			unsigned long pNextProcNumData = procNumDataBuffer[(p + 1 + maxProcs) % maxProcs];

			mpiSend(&pPrevProcNumData, 1, p);
			mpiSend(&pNextProcNumData, 1, p);
		}

		// classic permutation (not handling [3])
		p0_classic_permutation(maxProcs, dimension, procNumData, procNumDataBuffer, arows, bcols, result);
		
		// [3]. permutes accordingly to number of processors
        // if(procNumData > nextProcNumData){
        //     if(nextProcNumData == prevProcNumData)
		// 		// e.g. 3 2 2 (n = 7)
        //         // p0_permutation1(maxProcs, dimension, procNumData, nextProcNumData, prevProcNumData, arows, bcols, result);
        // } else if(procNumData == nextProcNumData){
        //     if(nextProcNumData > prevProcNumData)
		// 		// e.g. 3 3 2 (n = 8)
        //         // p0_permutation2(maxProcs, dimension, procNumData, nextProcNumData, prevProcNumData, arows, bcols, result);
        //     else if(nextProcNumData == prevProcNumData)
        //         // classic case where all processors have same proc num data
		// 		p0_classic_permutation(maxProcs, dimension, procNumData, procNumDataBuffer, arows, bcols, result);
        // }

		// if the processor handle only one row / column
		if(procNumData == 1){
            #pragma omp parallel for
			for(unsigned long i = 0; i < dimension; ++i)
				nset(C, (-i + dimension) % dimension, 0, nget(result, 0, i));
		} else{
			// for more
            #pragma omp parallel for
			for(unsigned long i = 0; i < procNumData; ++i){
				for(unsigned long j = 0; j < dimension; j += procNumData){
					long dataIndex = 0;

					if(j < procNumData)
						dataIndex = j;
					else if(j >= procNumData && j <= dimension - procNumData)
						dataIndex = dimension - j;
					else if(j == dimension - procNumData)
						dataIndex = procNumData + i;
					
					// fill the resulted matrix
					for(unsigned long dptr = 0; dptr < procNumData; ++dptr)
						nset(C, j + dptr, i, nget(result, i, dataIndex + dptr));
				}
			}
		}

		/* -- Performs the gather phase -- */
		for(unsigned long p = 1; p < maxProcs; ++p){

			// get the processors calculations results
			for(unsigned long idata = 0; idata < procNumDataBuffer[p]; ++idata)
				mpiRecv(ngetVec(result, idata), dimension, p);

			// if the processor handle only one row / column
			if(procNumData == 1){
                #pragma omp parallel for
				for(unsigned long i = 0; i < dimension; ++i)
					nset(C, (p - i + dimension) % dimension, p, nget(result, 0, i));
			} else {
				// for more
                #pragma omp parallel for
				for(unsigned long i = 0; i < procNumData; ++i){
					for(unsigned long j = 0; j < dimension; j += procNumData){
						long dataIndex = 0;

						if(j < procNumData)
							dataIndex = j;
						else if(j >= procNumData && j <= dimension - procNumData)
							dataIndex = dimension - j;
						else if(j == dimension - procNumData)
							dataIndex = procNumData + i;

						// computes the C row and column to fill
						// indices calculations were a bit tricky
						unsigned long row = ((p * procNumData) + j + dimension) % dimension;
						unsigned long col = p * procNumData + i;

						for(unsigned long dptr = 0; dptr < procNumData; ++dptr)
							nset(C, row + dptr, col, nget(result, i, dataIndex + dptr));
					}
				}
			}
		}

		// output the resulted matrix
		printMatrix(C);

        // release the memory
        free(procNumDataBuffer);
		destroyMatrix(B);
		destroyMatrix(A);
        destroyMatrix(C);
	}

	// [1..maxProcs - 1] processors
	else {
		// receiving initial data (a-rows and b-cols)
		for(unsigned long i = 0; i < procNumData; ++i){
			mpiRecv(ngetVec(arows, i), dimension, 0);
			mpiRecv(ngetVec(bcols, i), dimension, 0);
		}

		// initial multiplication calculation
        #pragma omp parallel for
		for(unsigned long i = 0; i < procNumData; ++i)
            for(unsigned long j = 0; j < procNumData; ++j)
 				nset(result, i, j, vdot(dimension, ngetVec(arows, j), ngetVec(bcols, i))); 

		// [3]. previous and next processors amount of data
		unsigned long prevProcNumData = 0;
        unsigned long nextProcNumData = 0;

		mpiRecv(&prevProcNumData, 1, 0);
		mpiRecv(&nextProcNumData, 1, 0);

		// classic permutation (not handling [3])
		pX_classic_permutation(rank, maxProcs, dimension, procNumData, arows, bcols, result);

		// if(rank != maxProcs - 1){
		// 	if(prevProcNumData > procNumData){
		// 		if(procNumData == nextProcNumData){
		// 			// not first and last procs, e.g. second proc 3 2 2 
		// 		} else if(procNumData > nextProcNumData){
		// 			// case that should not happen
		// 		}
		// 	} else if(prevProcNumData == procNumData){
		// 		if(procNumData == nextProcNumData){
		//			// classic case
		// 			pX_classic_permutation(rank, maxProcs, dimension, procNumData, arows, bcols, result);
		// 		}
		// 	}
		// } else {
		//		// last processor permutation handling
		// 		if(prevProcNumData > procNumData){
		// 			if(procNumData == nextProcNumData){
		// 			} else if(procNumData > nextProcNumData){
		// 				// I don't think this case can happen
		// 			} else if(procNumData < nextProcNumData){
		// 			}
		// 		} else if(prevProcNumData == procNumData){
		// 			if(procNumData == nextProcNumData){
		// 				pX_classic_permutation(rank, maxProcs, dimension, procNumData, arows, bcols, result);
		// 			} else if(procNumData < nextProcNumData){
		// 			}
		// 		}
		// 	}

		// Send all its results to the first processor
		for(unsigned long idata = 0; idata < procNumData; ++idata)
			mpiSend(ngetVec(result, idata), dimension, 0);
	}

	// all processors release their memory
	destroyMatrix(arows);
	destroyMatrix(bcols);
	destroyMatrix(result);
}


// [3] -------------------------------------------------------------------------

/* Some permutation code in order to achieve the third project part (fail) */

// void p0_permutation1(unsigned long maxProcs, unsigned long dimension, unsigned long procNumData, unsigned long nextProcNumData,
//         unsigned long prevProcNumData, struct matrix* arows, struct matrix* bcols, struct matrix* result){
	
//	// e.g. 3 2 2 (N == 7 ; P == 3), processor 0 work
//	// procNumData == nextProcNumData + 1
// 	// nextProcNumData == prevProcNumData
//	// procNumData == prevProcNumData + 1

// 	for(unsigned long iteration = 0; iteration < maxProcs; ++iteration){

// 		if(iteration < maxProcs - 1){
// 			// copy of vector to keep for now
// 			struct vector* vecPermuted = createVector(arows->numCols, ngetVec(arows, 0));

// 			// for each set of data
// 			for(unsigned long idata = 1; idata < procNumData; ++idata){

// 				mpiSend(ngetVec(arows, idata), dimension, 1);

// 				// Receive the last processor's row
// 				mpiRecv(ngetVec(arows, idata - 1), dimension, maxProcs - 1);

// 				// Performs the local dot calculation
// 				#pragma omp parallel for
// 				for(unsigned long ibcol = 0; ibcol < procNumData; ++ibcol){
// 					nset(result, ibcol, procNumData * (iteration + 1) + idata, vdot(dimension, ngetVec(arows, idata), ngetVec(bcols, ibcol)));
// 				}
// 			}
		
// 			nsetVec(arows, procNumData - 1, vecPermuted);
// 			destroyVector(vecPermuted);
// 		} else {
// 				// last permutation, all data should have been computed for this thread
// 				// send the remaining data
// 				mpiSend(ngetVec(arows, procNumData - 1), dimension, 1);
// 		}
// 	}
// }

// void p0_permutation2(unsigned long maxProcs, unsigned long dimension, unsigned long procNumData, unsigned long* procNumDataBuffer,
//         struct matrix* arows, struct matrix* bcols, struct matrix* result){
//	// e.g. 3 3 2 (N == 8 ; P == 3), processor 0 work
//	// procNumData == nextProcNumData
//	// nextProcNumData == prevProcNumData

// 	// Computes the rows permutation
// 	// There are (maxProcs - 1) permutations
// 	for(unsigned long iteration = 0; iteration < maxProcs - 1; ++iteration){

// 		// for each set of data (one row + one col)
// 		for(unsigned long idata = 0; idata < procNumDataBuffer[iteration]; ++idata){
// 			// Send the current first processor row to the second one
// 			mpiSend(ngetVec(arows, idata), dimension, 1);

// 			// Receive the last processor's row
// 			mpiRecv(ngetVec(arows, idata), dimension, maxProcs - 1);

// 			// Performs the local dot calculation
// 			#pragma omp parallel for
// 			for(unsigned long ibcol = 0; ibcol < procNumDataBuffer[iteration]; ++ibcol){
// 				nset(result, ibcol, procNumData * (iteration + 1) + idata, vdot(dimension, ngetVec(arows, idata), ngetVec(bcols, ibcol)));
// 			}
// 		}
// 	}
// }

// ---------------------------------------------------------------------------

// Thread 0 permutation (does not handle [3])
void p0_classic_permutation(unsigned long maxProcs, unsigned long dimension, unsigned long procNumData, unsigned long* procNumDataBuffer,
        struct matrix* arows, struct matrix* bcols, struct matrix* result){

	// Computes the rows permutation
	// There are (maxProcs - 1) permutations
	for(unsigned long iteration = 0; iteration < maxProcs - 1; ++iteration){

		// for each row
		for(unsigned long idata = 0; idata < procNumDataBuffer[iteration]; ++idata){
			// Send the current first processor row to the second one
			mpiSend(ngetVec(arows, idata), dimension, 1);

			// Receive the last processor's row
			mpiRecv(ngetVec(arows, idata), dimension, maxProcs - 1);

			// Performs the local dot calculation
			#pragma omp parallel for
			for(unsigned long ibcol = 0; ibcol < procNumDataBuffer[iteration]; ++ibcol){
				nset(result, ibcol, procNumData * (iteration + 1) + idata, vdot(dimension, ngetVec(arows, idata), ngetVec(bcols, ibcol)));
			}
		}
	}
}

// Thread != 0 permutation (does not handle [3])
void pX_classic_permutation(int rank, unsigned long maxProcs, unsigned long dimension, unsigned long procNumData, struct matrix* arows, struct matrix* bcols, struct matrix* result){
	unsigned long arow_ptr = 0;

	long* tmpArowData = malloc(sizeof(long) * dimension);

	// permutation, receive other a-row, computes locally
	// the dot and put it in the result vector before sending it
	// to thread 0
	for(unsigned long iteration = 0; iteration < maxProcs - 1; ++iteration){
		arow_ptr = 0;

		for(unsigned long idata = 0; idata < procNumData; ++idata){
			// Receive its new row
			mpiRecv(tmpArowData, dimension, (rank - 1 + maxProcs) % maxProcs);

			// Send its old row
			mpiSend(ngetVec(arows, arow_ptr), dimension, (rank + 1 + maxProcs) % maxProcs);

			nsetVecLight(arows, arow_ptr, tmpArowData, dimension);
			++arow_ptr;

			#pragma omp parallel for
			for(unsigned long ibcol = 0; ibcol < procNumData; ++ibcol){
				nset(result, ibcol, procNumData * (iteration + 1) + idata, vdot(dimension, ngetVec(arows, idata), ngetVec(bcols, ibcol)));
			}
		}
	}

	free(tmpArowData);
}

// "Normalised get, set"
long nget(struct matrix* mat, long row, long col){
	return mat->data[row * mat->numCols + col];
}

long* ngetVec(struct matrix* mat, long row){
    return &mat->data[row * mat->numCols];
}

void nset(struct matrix* mat, long row, long col, long value){
	mat->data[row * mat->numCols + col] = value;
}

void nsetVec(struct matrix* mat, long row, struct vector* vec){
    #pragma omp parallel for
    for(unsigned long i = 0; i < vec->dimension; ++i){
        mat->data[row * mat->numCols + i] = vec->data[i];
    }
}

void nsetVecLight(struct matrix* mat, long row, long* data, unsigned long size){
	#pragma omp parallel for
    for(unsigned long i = 0; i < size; ++i){
        mat->data[row * mat->numCols + i] = data[i];
    }
}

long vdot(unsigned long dimension, long* A_data, long* B_data){
    long dot = 0;
    unsigned long i;

    #pragma omp parallel for default(shared) private(i) reduction(+:dot)  
    for(i = 0; i < dimension; ++i){
        dot += A_data[i] * B_data[i];
    }

    return dot;
}

// Mpi wrappers

int mpiSend(const void* buf, int count, int dest){
	return MPI_Send(buf, count, MPI_LONG, dest, 0, MPI_COMM_WORLD);
}

int mpiRecv(void* buf, int count, int source){
	return MPI_Recv(buf, count, MPI_LONG, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


// Extraction

long* lightExtractRow(struct matrix* mat, unsigned long row){
	return &(mat->data[row * mat->numCols]);
}

struct vector* extractColumn(struct matrix* mat, unsigned long col){
	struct vector* vec = allocateVector(mat->numRows);
	for(unsigned long i = 0; i < vec->dimension; ++i)
		vec->data[i] = mat->data[i * mat->numCols + col];
	return vec;
}

// Allocations and destructions

struct matrix* allocateMatrix(unsigned long numRows, unsigned long numCols){
	struct matrix* mat = malloc(sizeof(struct matrix));
	mat->data = malloc(sizeof(long) * numRows * numCols);
	mat->numRows = numRows;
	mat->numCols = numCols;
	return mat;
}

struct vector* allocateVector(unsigned long dimension){
	struct vector* vec = malloc(sizeof(struct vector));
	vec->data = malloc(sizeof(long) * dimension);
	vec->dimension = dimension;
	return vec;
}

struct vector* createVector(unsigned long dimension, long* data){
	struct vector* vec = allocateVector(dimension);

    #pragma omp parallel for
	for(unsigned long i = 0; i < dimension; ++i)
		vec->data[i] = data[i];

	return vec;
}

void destroyMatrix(struct matrix* mat){
	free(mat->data);
	free(mat);
}

void destroyVector(struct vector* vec){
	free(vec->data);
	free(vec);
}

// Parsing

struct matrix* parse(const char* filePath){
	FILE* file = fopen(filePath, "r");

	// get size by parsing first line number of elements
	unsigned long size = 0;
	unsigned long ptr;

	// iterates one first time to obtain the matrix full size
	while(fscanf(file, "%ld", &ptr) == 1)
		size++;

	// as the matrix is squared, using sqrt
	// on the size gives the matrix's dimension
	unsigned long dimension = sqrt(size);
	struct matrix* src = allocateMatrix(dimension, dimension);

	ptr = 0;
	rewind(file);

	long tmp;
	unsigned long col = 0;
	unsigned long row = 0;

	// iterates a second time to obtain the matrix data
	while(ptr < size){
		if((ptr++) / (row + 1) >= dimension){
			col = 0;
			++row;
		}

		fscanf(file, "%ld", &tmp);
		nset(src, row, col++, tmp);
	}

	fclose(file);
	return src;
}

// Prints

void printMatrix(struct matrix* mat){
	for(unsigned long row = 0; row < mat->numRows; ++row){
		for(unsigned long col = 0; col < mat->numCols - 1; ++col){
			printf("%ld ", nget(mat, row, col));
		}

		printf("%ld\n", nget(mat, row, mat->numCols - 1));
	}
}