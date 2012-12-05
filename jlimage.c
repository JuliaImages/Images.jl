#include <stdlib.h>
#include <png.h>

// Compile with:
// gcc -fPIC -shared -L/home/tim/src/julia jlimage.c -o libjlimage.so -lpng -ljulia-release

// External declarations
void jl_error(const char *msg);

// Internal declarations
void jl_png_read_close(void **png_p);
void jl_png_write_close(void **png_p);



// Implementations
int jl_png_libpng_ver_major(void)
{
  return PNG_LIBPNG_VER_MAJOR;
}

int jl_png_libpng_ver_minor(void)
{
  return PNG_LIBPNG_VER_MINOR;
}

int jl_png_libpng_ver_release(void)
{
  return PNG_LIBPNG_VER_RELEASE;
}

void** jl_png_read_init(int fd)
{
  // We have to return multiple pointers. Pack them into a single
  // memory block.
  void **png_p = (void**) malloc(3*sizeof(void*));
  if (png_p == NULL)
    return NULL;
  png_p[0] = (void*) png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_p[0] == NULL) {
    free(png_p);
    return NULL;
  }
  png_p[1] = (void*) png_create_info_struct((png_structp)(png_p[0]));
  if (png_p[1] == NULL) {
    png_destroy_read_struct((png_structpp) png_p, (png_infopp)NULL, (png_infopp)NULL);
    free(png_p);
    return NULL;
  }
  // Open a traditional C file stream from the ios_t file descriptor
  png_p[2] = fdopen(fd, "r");
  if (png_p[2] == NULL) {
    png_destroy_read_struct((png_structpp) png_p, (png_infopp) (png_p+1), (png_infopp)NULL);
    free(png_p);
    return NULL;
  }
  // Set up the default error-handling scheme (to free the allocated memory)
  if (setjmp(((png_structp) (png_p[0]))->jmpbuf)) {
    jl_png_read_close(png_p);
    jl_errorf("PNG read error");
  }
  // Initialize I/O
  png_init_io((png_structp) (png_p[0]), (FILE*) (png_p[2]));

  return png_p;
}

int jl_png_read_image(png_structp png_ptr, png_infop  info_ptr, char *imbuf)
{
  png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
  png_uint_32 height = png_get_image_height(png_ptr, info_ptr);
  char **row_pointers = malloc(height*sizeof(char*));
  if (row_pointers == NULL)
    return -1;
  int i;
  for (i = 0; i < height; i++)
    row_pointers[i] = imbuf + i*((long) rowbytes);
  png_read_image(png_ptr, (png_bytepp) row_pointers);
  png_read_end(png_ptr, info_ptr);
  free(row_pointers);
  return 0;
}

void jl_png_read_close(void **png_p)
{
  if (png_p != NULL) {
    png_destroy_read_struct((png_structpp) png_p, (png_infopp) (png_p+1), (png_infopp)NULL);
    fclose(png_p[2]);
    free(png_p);
  }
}

void** jl_png_write_init(int fd)
{
  void **png_p = (void**) malloc(3*sizeof(void*));
  if (png_p == NULL)
    return NULL;
  png_p[0] = (void*) png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_p[0] == NULL) {
    free(png_p);
    return NULL;
  }
  png_p[1] = (void*) png_create_info_struct((png_structp)(png_p[0]));
  if (png_p[1] == NULL) {
    png_destroy_write_struct((png_structpp) png_p, (png_infopp)NULL);
    free(png_p);
    return NULL;
  }
  png_p[2] = fdopen(fd, "w");
  if (png_p[2] == NULL) {
    png_destroy_write_struct((png_structpp) png_p, (png_infopp) (png_p+1));
    free(png_p);
    return NULL;
  }
  // Set up the default error-handling scheme (to free the allocated memory)
  if (setjmp(((png_structp) (png_p[0]))->jmpbuf)) {
    jl_png_write_close(png_p);
    jl_errorf("PNG write error");
  }
  // Initialize I/O
  png_init_io((png_structp) (png_p[0]), (FILE*) (png_p[2]));

  return png_p;
}

int jl_png_write_image(png_structp png_ptr, png_infop  info_ptr, char *imbuf)
{
  png_uint_32 rowbytes = png_get_rowbytes(png_ptr, info_ptr);
  png_uint_32 height = png_get_image_height(png_ptr, info_ptr);
  char **row_pointers = malloc(height*sizeof(char*));
  if (row_pointers == NULL)
    return -1;
  int i;
  for (i = 0; i < height; i++)
    row_pointers[i] = imbuf + i*((long) rowbytes);
  png_write_image(png_ptr, (png_bytepp) row_pointers);
  png_write_end(png_ptr, info_ptr);
  free(row_pointers);
  return 0;
}

void jl_png_write_close(void **png_p)
{
  if (png_p != NULL) {
    png_destroy_write_struct((png_structpp) png_p, (png_infopp) (png_p+1));
    fclose(png_p[2]);
    free(png_p);
  }
}
