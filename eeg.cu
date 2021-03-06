/* EEG parsing application for 5SIA0
 *
 * Code by Glenn Bergmans (g.bergmans@student.tue.nl)
 * Code is based on various sources, most notably
 * the TU/e ES group code base and a Matlab
 * implementation by Mohammad Tahghighi
 */

#include "eeg.h"
#include <time.h>

int32_t randint(int32_t vmin, int32_t vmax)
{
    return (vmin + (int32_t) (rand() / (RAND_MAX / ((uint32_t) (vmax - vmin + 1)) + 1)));
}

int main(int argc, char *argv[]) {
	#ifndef CPU_ONLY
    // dummy function
    int32_t* dummy;
    cudaMallocManaged(&dummy, sizeof(int32_t));
    cudaFree(dummy);
	#endif

    #ifdef TIMING
    clock_t begin_tot = clock();
    #endif

    float features[CHANNELS][FEATURE_LENGTH];
    float favg[FEATURE_LENGTH] = {0};
    int32_t x[CHANNELS][DATAPOINTS];
	  uint32_t i, j;

    read_data(x, CHANNELS, DATAPOINTS);
    
    #ifdef TIMING_CSV
    printf("but,sta,p2p,ape,hur,pow;\r\n");
    #endif
    for (i = 0; i < CHANNELS; i++) {
        #ifdef VERBOSE
        printf("Running channel %d...\n", i);
        #endif
        run_channel(DATAPOINTS, x[i], features[i]);
    }

    // Averaging channels
    for (i = 0; i < CHANNELS; i++) {
        for (j = 0; j < FEATURE_LENGTH; j++) {
            favg[j] += features[i][j] / FEATURE_LENGTH;
        }
    }

    printf("\n");
	  for (i=0; i<FEATURE_LENGTH; i++)
        fprintf(stderr,"Feature %d: %.6f\n", i, favg[i]);

	  #ifdef TIMING
    clock_t end_tot = clock();
    double time_spent_tot = (double)(end_tot - begin_tot) / CLOCKS_PER_SEC;
    printf("Total time: %lfs\r\n", time_spent_tot);
	  
	//cuProfilerStop();
	  #endif
	  
    return 0;
}

void read_data(int32_t x[CHANNELS][DATAPOINTS], int nc, int np)
{
    FILE *fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    int l, c;

    fp = fopen("EEG.csv", "r");
    if (fp == NULL) {
        printf("Error opening EEG.csv\n");
        exit(EXIT_FAILURE);
    }

    // Skip the first line
    read = getline(&line, &len, fp);

    l = 0;
    while ((l < np) && (read = getline(&line, &len, fp)) != -1) {
        char *tok;
        tok = strtok(line, ",");
        float v;

        for (c = 0; c < nc; c++) {
            sscanf(tok, "%f", &v);
            x[c][l] = (int32_t) round(v);
            tok = strtok(NULL, ",");
        }

        l++;
    }

}

void run_channel(int np, int32_t *x, float *features)
{
	  #ifdef TIMING_ALL
    clock_t begin_but = clock();
	  #endif
	  
    // Butterworth returns np + 1 samples
	  int32_t *X = (int32_t *) malloc((np + 1) * sizeof(int32_t));

    // Clean signal using butterworth
    #ifdef VERBOSE
    printf("    Butterworth filter...\n");
    #endif
    bw0_int(np, x, X);

    #ifdef TIMING_ALL
	  clock_t end_but = clock();
    double time_but = (double)(end_but - begin_but) / CLOCKS_PER_SEC;
    #ifdef TIMING_CSV
    printf("%lf,", time_but);
    #else
    printf("Time but: %lfs\r\n", time_but);
    #endif
    clock_t begin_sta = clock();
	  #endif
    
		// 4 features: mean, std dev, abs sum, mean crossings
    #ifdef VERBOSE
    printf("    Standard features...\n");
    #endif
    stafeature(np, X, &features[0]);
    
		#ifdef TIMING_ALL
  	clock_t end_sta = clock();
    double time_sta = (double)(end_sta - begin_sta) / CLOCKS_PER_SEC;
    #ifdef TIMING_CSV
    printf("%lf,", time_sta);
    #else
    printf("Time sta: %lfs\r\n", time_sta);
    #endif
    clock_t begin_p2p = clock();
	  #endif
    
		// 2 features: mean p2p, std dev p2p
    #ifdef VERBOSE
    printf("    Peak 2 peak features...\n");
    #endif
    p2p(np, X, &features[4], 7);

	  #ifdef TIMING_ALL
	  clock_t end_p2p = clock();
    double time_p2p = (double)(end_p2p - begin_p2p) / CLOCKS_PER_SEC;
    #ifdef TIMING_CSV
    printf("%lf,", time_p2p);
    #else
    printf("Time p2p: %lfs\r\n", time_p2p);
    #endif
    clock_t begin_ape = clock();
  	#endif

    // 1 feature: aproximate entropy
    #ifdef VERBOSE
    printf("    Aproximate Entropy feature...\n");
    #endif
    apen(np, X, &features[6], 3, 0.2);

	  #ifdef TIMING_ALL
	  clock_t end_ape = clock();
    double time_ape = (double)(end_ape - begin_ape) / CLOCKS_PER_SEC;
    #ifdef TIMING_CSV
    printf("%lf,", time_ape);
    #else
    printf("Time p2p: %lfs\r\n", time_ape);
    #endif
    clock_t begin_hur = clock();
  	#endif

    // 1 feature: hurst coefficient
    #ifdef VERBOSE
    printf("    Hurst Coefficient feature...\n");
    #endif
    hurst(np, X, &features[7]);

	  #ifdef TIMING_ALL
	  clock_t end_hur = clock();
    double time_hur = (double)(end_hur - begin_hur) / CLOCKS_PER_SEC;
    #ifdef TIMING_CSV
    printf("%lf,", time_hur);
    #else
    printf("Time hur: %lfs\r\n", time_hur);
    #endif
    clock_t begin_pow = clock();
  	#endif

    // 6 features: power in 5 frequency bands & total power
    #ifdef VERBOSE
    printf("    Power Spectral Density features...\n");
    #endif
    power_per_band(np, X, &features[8]);

	  #ifdef TIMING_ALL
	  clock_t end_pow = clock();
    double time_pow = (double)(end_pow - begin_pow) / CLOCKS_PER_SEC;
    #ifdef TIMING_CSV
    printf("%lf;\r\n", time_pow);
    #else
    printf("Time pow: %lfs\r\n", time_pow);
  	#endif
    #endif

    #ifdef VERBOSE
    printf("Channel done\n");
    #endif
    free(X);
}
