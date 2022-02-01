// Melih Kurtaran (melih.kurtaran@abo.fi)

// In exercise 2, redesign the template for MPI: galaxy_mpi.c

// For MPI programs, compile with
//    mpicc -O3 -o galaxy_mpi galaxy_mpi.c -lm
//
// and run with e.g. 100 cores
//    srun -n 100 ./galaxy_mpi data_100k_arcmin.dat rand_100k_arcmin.dat omega.out    


// Uncomment as necessary
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float  pif;
long int MemoryAllocatedCPU = 0L;

long int NUM_OF_GALAXIES = 100000L;

int main(int argc, char* argv[]) 
{
    MPI_Status status;
    int parseargs_readinput(int argc, char *argv[]);
    int   id, ntasks;
    
    /* Initialize MPI */
    if ( MPI_Init(&argc, &argv) != MPI_SUCCESS) { //parallelization starts here
      printf("MPI_init failed!\n");
      exit(1);
    }

    /* Get the total number of tasks */
    if ( MPI_Comm_size(MPI_COMM_WORLD, &ntasks) != MPI_SUCCESS) {
      printf("MPI_Comm_size failed!\n");
      exit(1);
    }

    /* Get id of each processes */
    if ( MPI_Comm_rank(MPI_COMM_WORLD, &id) != MPI_SUCCESS) {
      printf("MPI_Comm_rank failed!\n");
      exit(1);
    }
	
    pif = acosf(-1.0f); // PI number

    // store right ascension and declination for real galaxies here
    // Note: indices run from 0 to 99999 = 100000-1: realrasc[0] -> realrasc[99999] 
    // realrasc[100000] is out of bounds for allocated memory!
    real_rasc        = (float *)calloc(NUM_OF_GALAXIES, sizeof(float));
    real_decl        = (float *)calloc(NUM_OF_GALAXIES, sizeof(float));

    // store right ascension and declination for synthetic random galaxies here
    rand_rasc        = (float *)calloc(NUM_OF_GALAXIES, sizeof(float));
    rand_decl        = (float *)calloc(NUM_OF_GALAXIES, sizeof(float));

    MemoryAllocatedCPU += 10L*NUM_OF_GALAXIES*sizeof(float);

    long int histogram_DD[360] = {0L};
    long int histogram_DR[360] = {0L};
    long int histogram_RR[360] = {0L};
    MemoryAllocatedCPU += 3L*360L*sizeof(long int);
    
    /* Process 0 (the receiver) does this */
    if (id == 0) {
        int source_id;
        struct timeval _ttime;
        struct timezone _tzone;
        gettimeofday(&_ttime, &_tzone);
        double time_start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
        
        /* Receive from all other processes */
        long int DD[360] = {0L};
        long int DR[360] = {0L};
        long int RR[360] = {0L};
        int i,j;
        
        if ( parseargs_readinput(argc, argv) != 0 ) {printf("   Program stopped.\n");return(0);}
        printf("   Input data read, now calculating histograms\n");
        
        // send the arrays to the other processes
        for (i=1; i<ntasks; i++) {
            MPI_Send(real_rasc, NUM_OF_GALAXIES, MPI_FLOAT, i, 21 , MPI_COMM_WORLD);
            MPI_Send(real_decl, NUM_OF_GALAXIES, MPI_FLOAT, i, 22 , MPI_COMM_WORLD);
            MPI_Send(rand_rasc, NUM_OF_GALAXIES, MPI_FLOAT, i, 23 , MPI_COMM_WORLD);
            MPI_Send(rand_decl, NUM_OF_GALAXIES, MPI_FLOAT, i, 24 , MPI_COMM_WORLD);
        }
        
        
        for (i=1; i<ntasks; i++) {
            if ( MPI_Recv(DD, 360, MPI_LONG, i, 0, MPI_COMM_WORLD,&status) != MPI_SUCCESS) {
                printf("Error in MPI_Recv!\n");
                exit(1);
            }
            for (j = 0; j < 360; ++j )
                histogram_DD[j] += DD[j];
        }
        for (i=1; i<ntasks; i++) {
            if ( MPI_Recv(DR, 360, MPI_LONG, i, 1, MPI_COMM_WORLD,&status) != MPI_SUCCESS) {
                printf("Error in MPI_Recv!\n");
                exit(1);
            }
            for (j = 0; j < 360; ++j )
                histogram_DR[j] += DR[j];
        }
        for (i=1; i<ntasks; i++) {
            if ( MPI_Recv(RR, 360, MPI_LONG, i, 2, MPI_COMM_WORLD,&status) != MPI_SUCCESS) {
                printf("Error in MPI_Recv!\n");
                exit(1);
            }
            for (j = 0; j < 360; ++j )
                histogram_RR[j] += RR[j];
        }
        
        // check point: the sum of all historgram entries should be 10 000 000 000 (NUM_OF_GALAXIES * NUM_OF_GALAXIES)
        long int histsum = 0L;
        int      correct_value=1;
        for ( int i = 0; i < 360; ++i ) histsum += histogram_DD[i];
        printf("   Histogram DD : sum = %ld\n",histsum);
        if ( histsum != NUM_OF_GALAXIES*NUM_OF_GALAXIES ) correct_value = 0;

        histsum = 0L;
        for ( int i = 0; i < 360; ++i ) histsum += histogram_DR[i];
        printf("   Histogram DR : sum = %ld\n",histsum);
        if ( histsum != NUM_OF_GALAXIES*NUM_OF_GALAXIES ) correct_value = 0;

        histsum = 0L;
        for ( int i = 0; i < 360; ++i ) histsum += histogram_RR[i];
        printf("   Histogram RR : sum = %ld\n",histsum);
        if ( histsum != NUM_OF_GALAXIES*NUM_OF_GALAXIES ) correct_value = 0;

        
        if ( correct_value != 1 )
        {
            printf("   Histogram sums should be 10000000000. Ending program prematurely\n");
            MPI_Finalize();
            return(0);
        }
         
        //Omega Calculation
        printf("   Omega values for the histograms:\n");
        float omega[360];
        for (i = 0; i < 360; ++i )
            if ( histogram_RR[i] != 0L )
               {
               omega[i] = (histogram_DD[i] - 2L*histogram_DR[i] + histogram_RR[i])/((float)(histogram_RR[i]));
               if ( i < 10 ) printf("      angle %.2f deg. -> %.2f deg. : %.3f\n", i*0.25, (i+1)*0.25, omega[i]);
               }

        FILE *out_file = fopen(argv[3],"w");
        if ( out_file == NULL ) printf("   ERROR: Cannot open output file %s\n",argv[3]);
        else
           {
           for (i = 0; i < 360; ++i )
               if ( histogram_RR[i] != 0L )
                  fprintf(out_file,"%.2f  : %.3f\n", i*0.25, omega[i] );
           fclose(out_file);
           printf("   Omega values written to file %s\n",argv[3]);
           }
           
        
        printf("   Total memory allocated = %.1lf MB\n",MemoryAllocatedCPU/1000000.0);
        gettimeofday(&_ttime, &_tzone);
        double time_end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

        printf("   Wall clock run time    = %.1lf secs\n",time_end - time_start);
    }

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

    // read input data from files given on the command line
    
    //all processes except the main one
    else {
        
        
        double angle; // angle between galaxies
        long int i,j;
        
        long int start = (NUM_OF_GALAXIES / (ntasks-1)) * (id-1); //start value for each process
        long int end; //end value for each process
        if (id != ntasks-1)
            end = (NUM_OF_GALAXIES / (ntasks-1)) * (id);
        else
            end = NUM_OF_GALAXIES;
        
        
        // printf("Start value of process %d: %d, end value: %d \n",id,start,end);
        
        //receive the array values from the main process
        MPI_Recv(real_rasc, NUM_OF_GALAXIES, MPI_FLOAT, 0, 21 , MPI_COMM_WORLD,&status);
        MPI_Recv(real_decl, NUM_OF_GALAXIES, MPI_FLOAT, 0, 22 , MPI_COMM_WORLD,&status);
        MPI_Recv(rand_rasc, NUM_OF_GALAXIES, MPI_FLOAT, 0, 23 , MPI_COMM_WORLD,&status);
        MPI_Recv(rand_decl, NUM_OF_GALAXIES, MPI_FLOAT, 0, 24 , MPI_COMM_WORLD,&status);
        
        
        // histogram_DD REAL TO REAL
        for(i=start;i<end;i++)
        {
            for(j=0;j<NUM_OF_GALAXIES;j++)
            {
                angle = 180/pif*acosf(sinf(real_decl[i])*sinf(real_decl[j]) + cosf(real_decl[i])*cosf(real_decl[j])*cosf(real_rasc[i]-real_rasc[j]));
                if(isnan(angle)) {angle = 0;}
                histogram_DD[ (int) floor(angle / 0.25) ]++;
            }
        }
        
        MPI_Send(histogram_DD, 360, MPI_LONG, 0, 0 , MPI_COMM_WORLD);
        
        // histogram_DR REAL TO RANDOM
        for(i=start;i<end;i++)
        {
            for(j=0;j<NUM_OF_GALAXIES;j++)
            {
                angle = 180/pif*acosf( sinf(real_decl[i])*sinf(rand_decl[j]) + cosf(real_decl[i])*cosf(rand_decl[j])*cosf(real_rasc[i]-rand_rasc[j]));
                if(isnan(angle)) {angle = 0;}
                histogram_DR[ (int) floor(angle / 0.25) ]++;
            }
        }
        
        MPI_Send(histogram_DR, 360, MPI_LONG, 0, 1 , MPI_COMM_WORLD);
        
        // histogram_RR RANDOM TO RANDOM
        for(i=start;i<end;i++)
        {
            for(j=0;j<NUM_OF_GALAXIES;j++)
            {
                angle = 180/pif*acosf(sinf(rand_decl[i])*sinf(rand_decl[j]) + cosf(rand_decl[i])*cosf(rand_decl[j])*cosf(rand_rasc[i]-rand_rasc[j]));
                if(isnan(angle)) {angle = 0;}
                histogram_RR[ (int) floor(angle / 0.25) ]++;
            }
        }
        
        MPI_Send(histogram_RR, 360, MPI_LONG, 0, 2 , MPI_COMM_WORLD);
    }
    
    free(real_rasc); free(real_decl);
    free(rand_rasc); free(rand_decl);
    
    // Terminate MPI execution environment
    if ( MPI_Finalize() != MPI_SUCCESS) {
      printf("Error in MPI_Finalize!\n");
      exit(1);
    }
    
    return(0);
}



int parseargs_readinput(int argc, char *argv[])
    {
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f/60.0f/180.0f*pif;
    int Number_of_Galaxies;
  
    if ( argc != 4 ) 
       {
       printf("   Usage: galaxy real_data random_data output_file\n   All MPI processes will be killed\n");
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
          if ( Number_of_Galaxies != 100000L )
             {
             printf("Cannot read file %s correctly, first item not 100000\n",argv[1]);
             fclose(real_data_file);
             return(1);
             } 
          for ( int i = 0; i < NUM_OF_GALAXIES; ++i )
              {
      	      float rasc, decl;
	      if ( fscanf(real_data_file,"%f %f", &rasc, &decl ) != 2 )
	         {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[1]);
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
          if ( Number_of_Galaxies != 100000L )
             {
             printf("Cannot read file %s correctly, first item not 100000\n",argv[2]);
             fclose(rand_data_file);
             return(1);
             } 
          for ( int i = 0; i < NUM_OF_GALAXIES; ++i )
              {
      	      float rasc, decl;
	      if ( fscanf(rand_data_file,"%f %f", &rasc, &decl ) != 2 )
	         {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[2]);
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

