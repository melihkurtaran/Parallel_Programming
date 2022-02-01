// Compilation on dione:
//    module load gcc               // do this once when you log in

// For OpemMP programs, compile with
//    gcc -O3 -fopenmp -o offload offload.c -lm
// and run with
//    srun -N 1 -c 40 ./offload RealGalaxies_100k_arcmin.dat SyntheticGalaxies_100k_arcmin.dat omega.out

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float  pif;
long int MemoryAllocatedCPU = 0L;

int main(int argc, char* argv[])
{
    /*
    argc = 4;
    argv[1] = "RealGalaxies_100k_arcmin.txt";
    argv[2] = "SyntheticGalaxies_100k_arcmin.txt";
    argv[3] = "omega.out";
     */
    
    int parseargs_readinput(int argc, char *argv[]);
    struct timeval _ttime;
    struct timezone _tzone;

    pif = acosf(-1.0f);

    gettimeofday(&_ttime, &_tzone);
    double time_start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    // store right ascension and declination for real galaxies here
    // Note: indices run from 0 to 99999 = 100000-1: realrasc[0] -> realrasc[99999]
    // realrasc[100000] is out of bounds for allocated memory!
    real_rasc        = (float *)calloc(100000L, sizeof(float));
    real_decl        = (float *)calloc(100000L, sizeof(float));

    // store right ascension and declination for synthetic random galaxies here
    rand_rasc        = (float *)calloc(100000L, sizeof(float));
    rand_decl        = (float *)calloc(100000L, sizeof(float));
    
    MemoryAllocatedCPU += 10L*100000L*sizeof(float);

    // read input data from files given on the command line
    if ( parseargs_readinput(argc, argv) != 0 ) {printf("   Program stopped.\n");return(0);}
    printf("   Input data read, now calculating histograms\n");

    long int histogram_DD[360] = {0L};
    long int histogram_DR[360] = {0L};
    long int histogram_RR[360] = {0L};
    MemoryAllocatedCPU += 3L*360L*sizeof(long int);
    
//  Your code to calculate angles and filling the histograms
//  helpful hint: there are no angles above 90 degrees!
//  histogram[0] covers  0.00 <=  angle  <   0.25
//  histogram[1] covers  0.25 <=  angle  <   0.50
//  histogram[2] covers  0.50 <=  angle  <   0.75
//  histogram[3] covers  0.75 <=  angle  <   1.00
//  histogram[4] covers  1.00 <=  angle  <   1.25
//  and so on until     89.75 <=  angle  <= 90.0

    // here goes your code to calculate angles and to fill in
    // histogram_DD, histogram_DR and histogram_RR
    // use as input data the arrays real_rasc[], real_decl[], rand_rasc[], rand_decl[]

    double angle; // angle between galaxies
    long int i,j;
    
    // histogram_DD REAL TO REAL

    #pragma omp target map(to: real_decl[0:100000], real_rasc[0:100000]) map(tofrom: histogram_DD[0:360])
    for(i=0;i<100000L;i+=100)
    {
        for(j=0;j<100000L;j+=100)
        {
            angle = 180/pif*acosf(sinf(real_decl[i])*sinf(real_decl[j]) + cosf(real_decl[i])*cosf(real_decl[j])*cosf(real_rasc[i]-real_rasc[j]));
            if(isnan(angle)) {angle = 0;}
            #pragma omp atomic
            histogram_DD[ (int) floor(angle / 0.25) ]++;
        }
    }
    
    // histogram_DR REAL TO RANDOM
    #pragma omp target map(to: real_decl[0:100000], real_rasc[0:100000],rand_decl[0:100000], rand_rasc[0:100000]) map(tofrom: histogram_DD[0:360])
    for(i=0;i<100000L;i+=100)
    {
        for(j=0;j<100000L;j+=100)
        {
            angle = 180/pif*acosf( sinf(real_decl[i])*sinf(rand_decl[j]) + cosf(real_decl[i])*cosf(rand_decl[j])*cosf(real_rasc[i]-rand_rasc[j]));
            if(isnan(angle)) {angle = 0;}
            #pragma omp atomic
            histogram_DR[ (int) floor(angle / 0.25) ]++;
        }
    }
    
    // histogram_RR RANDOM TO RANDOM
    #pragma omp target map(to: rand_decl[0:100000], rand_rasc[0:100000]) map(tofrom: histogram_RR[0:360])
    for(i=0;i<100000L;i+=100)
    {
        for(j=0;j<100000L;j+=100)
        {
            angle = 180/pif*acosf(sinf(rand_decl[i])*sinf(rand_decl[j]) + cosf(rand_decl[i])*cosf(rand_decl[j])*cosf(rand_rasc[i]-rand_rasc[j]));
            if(isnan(angle)) {angle = 0;}
            #pragma omp atomic
            histogram_RR[ (int) floor(angle / 0.25) ]++;
        }
    }



    // check point: the sum of all historgram entries should be 10 000 000 000
    long int histsum = 0L;
    int      correct_value=1;
    for ( i = 0; i < 360; ++i ) histsum += histogram_DD[i];
    printf("   Histogram DD : sum = %ld\n",histsum);
    if ( histsum != 10000000000L ) correct_value = 0;

    histsum = 0L;
    for ( i = 0; i < 360; ++i ) histsum += histogram_DR[i];
    printf("   Histogram DR : sum = %ld\n",histsum);
    if ( histsum != 10000000000L ) correct_value = 0;

    histsum = 0L;
    for ( i = 0; i < 360; ++i ) histsum += histogram_RR[i];
    printf("   Histogram RR : sum = %ld\n",histsum);
    if ( histsum != 10000000000L ) correct_value = 0;

//    if ( correct_value != 1 )
//       {printf("   Histogram sums should be 10000000000. Ending program prematurely\n");return(0);}

    printf("   Omega values for the histograms:\n");
    float omega[360];
    for ( i = 0; i < 360; ++i )
        if ( histogram_RR[i] != 0L )
           {
           omega[i] = (histogram_DD[i] - 2L*histogram_DR[i] + histogram_RR[i])/((float)(histogram_RR[i]));
           if ( i < 5 ) printf("      angle %.2f deg. -> %.2f deg. : %.3f\n", i*0.25, (i+1)*0.25, omega[i]);
           }

    FILE *out_file = fopen(argv[3],"w");
    if ( out_file == NULL ) printf("   ERROR: Cannot open output file %s\n",argv[3]);
    else
       {
       for ( i = 0; i < 360; ++i )
           if ( histogram_RR[i] != 0L )
              fprintf(out_file,"%.2f  : %.3f\n", i*0.25, omega[i] );
       fclose(out_file);
       printf("   Omega values written to file %s\n",argv[3]);
       }
       

    free(real_rasc); free(real_decl);
    free(rand_rasc); free(rand_decl);

    printf("   Total memory allocated = %.1lf MB\n",MemoryAllocatedCPU/1000000.0);
    gettimeofday(&_ttime, &_tzone);
    double time_end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

    printf("   Wall clock run time    = %.1lf secs\n",time_end - time_start);

    return(0);
}



int parseargs_readinput(int argc, char *argv[])
{
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f/60.0f/180.0f*pif;
    int Number_of_Galaxies;
    long int i = 0;
  
    if ( argc != 4 )
    {
        printf("%d",argc);
        printf("   Usage: galaxy real_data random_data output_file\n   All OpenMP processes will be killed\n");
        return(1);
    }
    if ( argc == 4 )
    {
       printf("   Running galaxy_openmp %s %s %s\n",argv[1], argv[2], argv[3]);

       real_data_file = fopen(argv[1],"r");
       if ( real_data_file == NULL )
        {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open real data file %s\n",argv[1]);
          return(1);
        }
       else
      {
          fscanf(real_data_file,"%d",&Number_of_Galaxies);
          for (i = 0; i < 100000; ++i )
              {
                float rasc, decl;
          if ( fscanf(real_data_file,"%f %f", &rasc, &decl ) != 2 )
             {
                 printf("   ERROR: Cannot read line %ld in real data file %s\n",i+1,argv[1]);
                 fclose(real_data_file);
             return(1);
             }
          real_rasc[i] = rasc*arcmin2rad;
          real_decl[i] = decl*arcmin2rad;
          }
           fclose(real_data_file);
       printf("   Successfully read 100000 lines from %s\n",argv[1]);
       }

       rand_data_file = fopen(argv[2],"r");
       if ( rand_data_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open random data file %s\n",argv[2]);
          return(1);
          }
       else
      {
          fscanf(rand_data_file,"%d",&Number_of_Galaxies);
          for ( i = 0; i < 100000; ++i )
              {
                float rasc, decl;
          if ( fscanf(rand_data_file,"%f %f", &rasc, &decl ) != 2 )
             {
                 printf("   ERROR: Cannot read line %ld in real data file %s\n",i+1,argv[2]);
                 fclose(rand_data_file);
             return(1);
             }
          rand_rasc[i] = rasc*arcmin2rad;
          rand_decl[i] = decl*arcmin2rad;
          }
          fclose(rand_data_file);
      printf("   Successfully read 100000 lines from %s\n",argv[2]);
      }
       out_file = fopen(argv[3],"w");
       if ( out_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open output file %s\n",argv[3]);
          return(1);
          }
       else fclose(out_file);
       }

    return(0);
    }




