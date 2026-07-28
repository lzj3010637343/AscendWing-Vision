/**
 * ascend_shm.c — POSIX 共享内存 IPC
 * Writer: memcpy → sem_post | Reader: sem_wait → memcpy
 */
#include "ascend_shm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>

#define POSIX_DET_SHM "/detection_shm"
#define POSIX_DET_SEM "/detection_sem"
#define POSIX_IMG_SHM "/image_shm"
#define POSIX_IMG_SEM "/image_sem"

static int g_bench=0; static char g_label[32]="ashm";
static inline uint64_t us(void){struct timeval tv;gettimeofday(&tv,NULL);return (uint64_t)tv.tv_sec*1000000ULL+tv.tv_usec;}
#define BLOG(f,...) do{if(g_bench)fprintf(stderr,"[%s] " f "\n",g_label,##__VA_ARGS__);}while(0)
void aclShmBenchmarkEnable(int e){g_bench=e;}
void aclShmBenchmarkSetLabel(const char *l){if(l){strncpy(g_label,l,sizeof(g_label)-1);g_label[sizeof(g_label)-1]=0;}}

static struct{int init,writer,fd_det,fd_img;void *p_det,*p_img;size_t sz_det,sz_img;sem_t *sem_det,*sem_img;}
g={0,0,-1,-1,NULL,NULL,0,0,SEM_FAILED,SEM_FAILED};

static int sc(const char*n,size_t s,int*fd,void**p){
 int f=shm_open(n,O_CREAT|O_RDWR,0666);if(f<0)return ASHM_ERR;
 struct stat st;if(fstat(f,&st)!=0||(size_t)st.st_size<s)ftruncate(f,s);
 void*pt=mmap(NULL,s,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
 if(pt==MAP_FAILED){close(f);return ASHM_ERR;}*fd=f;*p=pt;return ASHM_OK;
}
static int so(const char*n,size_t s,int*fd,void**p){
 int f=shm_open(n,O_RDWR,0666);if(f<0)return ASHM_ERR;
 void*pt=mmap(NULL,s,PROT_READ|PROT_WRITE,MAP_SHARED,f,0);
 if(pt==MAP_FAILED){close(f);return ASHM_ERR;}*fd=f;*p=pt;return ASHM_OK;
}
static sem_t*mks(const char*n){sem_unlink(n);return sem_open(n,O_CREAT|O_EXCL,0666,0);}
static sem_t*ops(const char*n){return sem_open(n,0);}

int aclShmWriterInit(void){
 if(g.init)return ASHM_OK;
 if(sc(POSIX_DET_SHM,sizeof(AShmDetectionFrame),&g.fd_det,&g.p_det)!=ASHM_OK)return ASHM_ERR;
 g.sz_det=sizeof(AShmDetectionFrame);
 if(sc(POSIX_IMG_SHM,ASHM_MAX_JPEG_SIZE+4,&g.fd_img,&g.p_img)!=ASHM_OK)return ASHM_ERR;
 g.sz_img=ASHM_MAX_JPEG_SIZE+4;
 g.sem_det=mks(POSIX_DET_SEM);g.sem_img=mks(POSIX_IMG_SEM);
 if(g.sem_det==SEM_FAILED||g.sem_img==SEM_FAILED)return ASHM_ERR;
 g.init=1;g.writer=1;
 fprintf(stderr,"[ashm] POSIX shm writer ready (det=%zuB img=%zuB)\n",g.sz_det,g.sz_img);
 return ASHM_OK;
}
int aclShmPutDetections(const AShmDetectionFrame*f){
 if(!g.init||!g.writer)return ASHM_ERR;
 memcpy(g.p_det,f,sizeof(AShmDetectionFrame));sem_post(g.sem_det);return ASHM_OK;
}
int aclShmPutImage(const uint8_t*d,uint32_t sz){
 if(!g.init||!g.writer)return ASHM_ERR;
 if(sz>ASHM_MAX_JPEG_SIZE)return ASHM_ERR;
 uint32_t*h=(uint32_t*)g.p_img;*h=sz;
 memcpy((uint8_t*)g.p_img+4,d,sz);sem_post(g.sem_img);return ASHM_OK;
}
void aclShmWriterCleanup(void){
 if(!g.init||!g.writer)return;
 if(g.p_det){munmap(g.p_det,g.sz_det);shm_unlink(POSIX_DET_SHM);}
 if(g.p_img){munmap(g.p_img,g.sz_img);shm_unlink(POSIX_IMG_SHM);}
 if(g.sem_det!=SEM_FAILED){sem_close(g.sem_det);sem_unlink(POSIX_DET_SEM);}
 if(g.sem_img!=SEM_FAILED){sem_close(g.sem_img);sem_unlink(POSIX_IMG_SEM);}
 g.init=0;fprintf(stderr,"[ashm] Writer cleanup done\n");
}

int aclShmReaderInit(const char*q){
 if(g.init)return ASHM_OK;
 int d=(strcmp(q,ASHM_DETECTION_Q_NAME)==0);
 if(d){
  if(so(POSIX_DET_SHM,sizeof(AShmDetectionFrame),&g.fd_det,&g.p_det)!=ASHM_OK)return ASHM_ERR;
  g.sem_det=ops(POSIX_DET_SEM);if(g.sem_det==SEM_FAILED)return ASHM_ERR;
  g.sz_det=sizeof(AShmDetectionFrame);fprintf(stderr,"[ashm] det reader ok\n");
 }else{
  if(so(POSIX_IMG_SHM,ASHM_MAX_JPEG_SIZE+4,&g.fd_img,&g.p_img)!=ASHM_OK)return ASHM_ERR;
  g.sem_img=ops(POSIX_IMG_SEM);if(g.sem_img==SEM_FAILED)return ASHM_ERR;
  g.sz_img=ASHM_MAX_JPEG_SIZE+4;fprintf(stderr,"[ashm] img reader ok\n");
 }
 g.init=1;g.writer=0;return ASHM_OK;
}
int aclShmGetDetections(AShmDetectionFrame*f,int to){
 if(!g.init||g.writer)return ASHM_ERR;
 uint64_t t0=us();
 int r;if(to<0)r=sem_wait(g.sem_det);else{struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);
  ts.tv_sec+=to/1000;ts.tv_nsec+=(to%1000)*1000000L;
  if(ts.tv_nsec>=1000000000L){ts.tv_sec++;ts.tv_nsec-=1000000000L;}r=sem_timedwait(g.sem_det,&ts);}
 if(r!=0)return(errno==ETIMEDOUT)?ASHM_TIMEOUT:ASHM_ERR;
 uint64_t t1=us();
 memcpy(f,g.p_det,sizeof(AShmDetectionFrame));
 uint64_t t2=us();
 BLOG("GetDet %u objs | sem=%lluus copy=%lluus total=%lluus",
  f->num_detections,(unsigned long long)(t1-t0),(unsigned long long)(t2-t1),(unsigned long long)(t2-t0));
 return ASHM_OK;
}
int aclShmGetImage(uint8_t*b,uint32_t bs,uint32_t*sz,int to){
 if(!g.init||g.writer)return ASHM_ERR;
 uint64_t t0=us();
 int r;if(to<0)r=sem_wait(g.sem_img);else{struct timespec ts;clock_gettime(CLOCK_REALTIME,&ts);
  ts.tv_sec+=to/1000;ts.tv_nsec+=(to%1000)*1000000L;
  if(ts.tv_nsec>=1000000000L){ts.tv_sec++;ts.tv_nsec-=1000000000L;}r=sem_timedwait(g.sem_img,&ts);}
 if(r!=0)return(errno==ETIMEDOUT)?ASHM_TIMEOUT:ASHM_ERR;
 uint64_t t1=us();
 uint32_t s=*(uint32_t*)g.p_img;if(s>bs)s=bs;
 memcpy(b,(uint8_t*)g.p_img+4,s);*sz=s;
 uint64_t t2=us();
 BLOG("GetImg %u B | sem=%lluus copy=%lluus total=%lluus",
  s,(unsigned long long)(t1-t0),(unsigned long long)(t2-t1),(unsigned long long)(t2-t0));
 return ASHM_OK;
}
void aclShmReaderCleanup(void){
 if(!g.init||g.writer)return;
 if(g.p_det){munmap(g.p_det,g.sz_det);close(g.fd_det);}
 if(g.p_img){munmap(g.p_img,g.sz_img);close(g.fd_img);}
 if(g.sem_det!=SEM_FAILED)sem_close(g.sem_det);
 if(g.sem_img!=SEM_FAILED)sem_close(g.sem_img);
 g.init=0;fprintf(stderr,"[ashm] Reader cleanup done\n");
}
