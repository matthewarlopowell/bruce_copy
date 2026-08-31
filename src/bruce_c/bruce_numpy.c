#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <omp.h>

#include "bruce_src.h"

#define N_BIN_MAX 10000

// loglike function implementation with OpenMP
static PyObject* loglike(PyObject* self, PyObject* args) {
    PyArrayObject *y = NULL, *yerr = NULL, *model = NULL;
    //PyArrayObject *y_cast = NULL, *yerr_cast = NULL, *model_cast = NULL;
    double jitter, offset;


    // Parse Python arguments (y, yerr, model, jitter, offset)
    if (!PyArg_ParseTuple(args, "O!O!O!dd", 
                          &PyArray_Type, &y, 
                          &PyArray_Type, &yerr, 
                          &PyArray_Type, &model, 
                          &jitter, &offset)) {
        return NULL;
    }

    // // Ensure the arrays are cast to double if necessary
    // if (PyArray_TYPE(y) != NPY_DOUBLE) {
    //     y_cast = (PyArrayObject*) PyArray_FROMANY((PyObject*) y, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
    //     if (!y_cast) goto fail;
    //     y = y_cast; // Use the casted array
    // }

    // Ensure the y array is of type double and contiguous
    PyArrayObject *y_cast = (PyArrayObject *)PyArray_FromAny(
    (PyObject *)y, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (y_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'y' to numpy.float64.");
        return NULL;
    }

    // if (PyArray_TYPE(yerr) != NPY_DOUBLE) {
    //     yerr_cast = (PyArrayObject*) PyArray_FROMANY((PyObject*) yerr, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
    //     if (!yerr_cast) goto fail;
    //     yerr = yerr_cast;
    // }

    // Ensure the y array is of type double and contiguous
    PyArrayObject *yerr_cast = (PyArrayObject *)PyArray_FromAny(
    (PyObject *)yerr, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (yerr_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'yerr' to numpy.float64.");
        return NULL;
    }

    // if (PyArray_TYPE(model) != NPY_DOUBLE) {
    //     model_cast = (PyArrayObject*) PyArray_FROMANY((PyObject*) model, NPY_DOUBLE, 0, 0, NPY_ARRAY_IN_ARRAY);
    //     if (!model_cast) goto fail;
    //     model = model_cast;
    // }

    // Ensure the y array is of type double and contiguous
    PyArrayObject *model_cast = (PyArrayObject *)PyArray_FromAny(
    (PyObject *)model, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (model_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'yerr' to numpy.float64.");
        return NULL;
    }

    // Check that the arrays are contiguous
    if (!PyArray_ISCONTIGUOUS(y) || !PyArray_ISCONTIGUOUS(yerr) || !PyArray_ISCONTIGUOUS(model)) {
        PyErr_SetString(PyExc_TypeError, "All input arrays must be contiguous.");
        goto fail;
    }

    // Check that the arrays have the same size
    npy_intp size = PyArray_SIZE(y);
    if (size != PyArray_SIZE(yerr) || size != PyArray_SIZE(model)) {
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same size.");
        goto fail;
    }

    // Access the data of the input arrays
    double *y_data = (double*) PyArray_DATA(y);
    double *yerr_data = (double*) PyArray_DATA(yerr);
    double *model_data = (double*) PyArray_DATA(model);

    // Initialize log-likelihood
    double loglike_val = 0.0;

    // OpenMP parallel reduction
    #pragma omp parallel for reduction(-:loglike_val)
    for (npy_intp i = 0; i < size; i++) {
        loglike_val -= bruce_loglike(y_data[i], yerr_data[i], model_data[i], jitter, offset);
    }

    // Free casted arrays (if allocated) and return result
    Py_XDECREF(y_cast);
    Py_XDECREF(yerr_cast);
    Py_XDECREF(model_cast);
    return PyFloat_FromDouble(loglike_val);

fail:
    Py_XDECREF(y_cast);
    Py_XDECREF(yerr_cast);
    Py_XDECREF(model_cast);
    return NULL;
}








// Python wrapper for the _lc function
static PyObject* lc(PyObject* self, PyObject* args) {
    PyArrayObject *time_array = NULL, *output_array = NULL;
    double t_zero, period, radius_1, k, incl, e, w, c, alpha, cadence, light_3;
    int noversample, ld_law, accurate_tp;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!ddddddddddidii", //"O!ddddddddddiidi",
                          &PyArray_Type, &time_array,  // NumPy array for time
                          &t_zero, &period, &radius_1, &k,
                          &incl, &e, &w, &c, &alpha, &cadence,
                          &noversample, &light_3, &ld_law, &accurate_tp)) {
        return NULL;  // Parsing failed
    }

    // Ensure the time array is of type double and contiguous
    PyArrayObject *time_cast = (PyArrayObject *)PyArray_FromAny(
    (PyObject *)time_array, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (time_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'time_array' to numpy.float64.");
        return NULL;
    }

    // printf("t_zero C code: %f\n", t_zero);
    // fflush(stdout);

    // PyArrayObject *time_cast = NULL;
    // if (PyArray_TYPE(time_array) != NPY_DOUBLE) {
    //     time_cast = (PyArrayObject*) PyArray_FROMANY(
    //         (PyObject*) time_array, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    //     if (!time_cast) {
    //         PyErr_SetString(PyExc_TypeError, "Failed to cast time array to numpy.float64.");
    //         return NULL;
    //     }
    // } else {
    //     time_cast = time_array;
    //     Py_INCREF(time_cast);  // Increment reference since we won't own it otherwise
    // }

    // Get the size of the time array
    npy_intp size = PyArray_SIZE(time_cast);

    // Create an output array of the same size as the time array
    output_array = (PyArrayObject*) PyArray_SimpleNew(1, PyArray_DIMS(time_cast), NPY_DOUBLE);
    if (!output_array) {
        Py_DECREF(time_cast);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array.");
        return NULL;
    }

    // Access the data pointers for the arrays
    double *time_data = (double*) PyArray_DATA(time_cast);
    double *output_data = (double*) PyArray_DATA(output_array);

    // Perform computation for each element in the time array
    #pragma omp parallel for
    for (npy_intp i = 0; i < size; i++) {
        output_data[i] = _lc(time_data[i], t_zero, period, radius_1, k,
                             incl, e, w, c, alpha, cadence,
                             noversample, light_3, ld_law, accurate_tp);
    }

    // Clean up and return the result
    Py_DECREF(time_cast);
    return (PyObject*) output_array;
}













static PyObject* lc_loglike(PyObject* self, PyObject* args) {
    PyArrayObject *time_array = NULL, *flux_array = NULL, *flux_err_array = NULL;
    double t_zero, period, radius_1, k, incl, e, w, c, alpha, cadence, light_3, jitter, offset;
    int noversample, ld_law, accurate_tp;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "O!O!O!ddddddddddidiidi",
                          &PyArray_Type, &time_array,    // NumPy array for time
                          &PyArray_Type, &flux_array,    // NumPy array for flux
                          &PyArray_Type, &flux_err_array, // NumPy array for flux error
                          &t_zero, &period, &radius_1, &k,
                          &incl, &e, &w, &c, &alpha, &cadence,
                          &noversample, &light_3, &ld_law, &accurate_tp,
                          &jitter, &offset)) {
        return NULL;  // Parsing failed
    }

    // Ensure the input arrays are of type double and contiguous
    // PyArrayObject *time_cast = NULL, *flux_cast = NULL, *flux_err_cast = NULL;
    // if (PyArray_TYPE(time_array) != NPY_DOUBLE) {
    //     time_cast = (PyArrayObject*) PyArray_FROMANY(
    //         (PyObject*) time_array, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    //     if (!time_cast) {
    //         PyErr_SetString(PyExc_TypeError, "Failed to cast time array to numpy.float64.");
    //         return NULL;
    //     }
    // } else {
    //     time_cast = time_array;
    //     Py_INCREF(time_cast);
    // }

        // Ensure the time array is of type double and contiguous
    PyArrayObject *time_cast = (PyArrayObject *)PyArray_FromAny(
    (PyObject *)time_array, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (time_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'time_array' to numpy.float64.");
        return NULL;
    }

    // if (PyArray_TYPE(flux_array) != NPY_DOUBLE) {
    //     flux_cast = (PyArrayObject*) PyArray_FROMANY(
    //         (PyObject*) flux_array, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    //     if (!flux_cast) {
    //         Py_DECREF(time_cast);
    //         PyErr_SetString(PyExc_TypeError, "Failed to cast flux array to numpy.float64.");
    //         return NULL;
    //     }
    // } else {
    //     flux_cast = flux_array;
    //     Py_INCREF(flux_cast);
    // }

    // Ensure the flux array is of type double and contiguous
    PyArrayObject *flux_cast = (PyArrayObject *)PyArray_FromAny(
    (PyObject *)flux_array, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (flux_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'flux_array' to numpy.float64.");
        return NULL;
    }

    // if (PyArray_TYPE(flux_err_array) != NPY_DOUBLE) {
    //     flux_err_cast = (PyArrayObject*) PyArray_FROMANY(
    //         (PyObject*) flux_err_array, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    //     if (!flux_err_cast) {
    //         Py_DECREF(time_cast);
    //         Py_DECREF(flux_cast);
    //         PyErr_SetString(PyExc_TypeError, "Failed to cast flux error array to numpy.float64.");
    //         return NULL;
    //     }
    // } else {
    //     flux_err_cast = flux_err_array;
    //     Py_INCREF(flux_err_cast);
    // }

    // Ensure the flux array is of type double and contiguous
    PyArrayObject *flux_err_cast = (PyArrayObject *)PyArray_FromAny(
    (PyObject *)flux_err_array, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (flux_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'flux_err_array' to numpy.float64.");
        return NULL;
    }

    // Ensure arrays have the same size
    npy_intp size = PyArray_SIZE(time_cast);
    if (size != PyArray_SIZE(flux_cast) || size != PyArray_SIZE(flux_err_cast)) {
        Py_DECREF(time_cast);
        Py_DECREF(flux_cast);
        Py_DECREF(flux_err_cast);
        PyErr_SetString(PyExc_ValueError, "Input arrays must have the same size.");
        return NULL;
    }

    // Access the data pointers for the arrays
    double *time_data = (double*) PyArray_DATA(time_cast);
    double *flux_data = (double*) PyArray_DATA(flux_cast);
    double *flux_err_data = (double*) PyArray_DATA(flux_err_cast);

    // Initialize log-likelihood
    double loglike_val = 0.0;

    // Perform computation in parallel using OpenMP
    #pragma omp parallel for reduction(-:loglike_val)
    for (npy_intp i = 0; i < size; i++) {
        // Compute the light curve model for this index
        double model = _lc(time_data[i], t_zero, period, radius_1, k,
                           incl, e, w, c, alpha, cadence,
                           noversample, light_3, ld_law, accurate_tp);
        loglike_val -= bruce_loglike(flux_data[i], flux_err_data[i], model, jitter, offset);
    }

    // Clean up and return the result
    Py_DECREF(time_cast);
    Py_DECREF(flux_cast);
    Py_DECREF(flux_err_cast);
    return PyFloat_FromDouble(loglike_val);
}










// static void print_array_max(PyArrayObject *array) {
//     // Ensure the input is an array
//     if (!PyArray_Check(array)) {
//         printf("Input is not a valid NumPy array.\n");
//         return;
//     }

//     // Get the maximum value using PyArray_Max
//     PyObject *max_value = PyArray_Max((PyObject *)array, NPY_MAXDIMS, NULL);  // NPY_MAXDIMS for the entire array
//     if (!max_value) {
//         PyErr_Print();  // Print any Python error that occurred
//         return;
//     }

//     // Print the maximum value
//     PyObject *max_value_str = PyObject_Str(max_value);
//     if (max_value_str) {
//         printf("Maximum value: %s\n", PyUnicode_AsUTF8(max_value_str));
//         Py_DECREF(max_value_str);
//     }

//     // Clean up
//     Py_DECREF(max_value);
// }





static PyObject *bin_data(PyObject *self, PyObject *args) {
    PyArrayObject *time_array, *flux_array;
    double bin_size;
    if (!PyArg_ParseTuple(args, "O!O!d",
                          &PyArray_Type, &time_array,
                          &PyArray_Type, &flux_array,
                          &bin_size)) {
        return NULL;
    }

    // if (PyArray_TYPE(time_array) != NPY_DOUBLE || PyArray_TYPE(flux_array) != NPY_DOUBLE) {
    //     PyErr_SetString(PyExc_TypeError, "Input arrays must be of type numpy.float64.");
    //     return NULL;
    // }
    // Ensure input arrays are cast to NPY_DOUBLE
    PyArrayObject *time_array_cast = (PyArrayObject *)PyArray_FromAny(
        (PyObject *)time_array, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (time_array_cast == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'time_array' to numpy.float64.");
        return NULL;
    }

    PyArrayObject *flux_array_cast = (PyArrayObject *)PyArray_FromAny(
        (PyObject *)flux_array, PyArray_DescrFromType(NPY_DOUBLE), 0, 0, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS, NULL);
    if (flux_array_cast == NULL) {
        Py_DECREF(time_array_cast);
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'flux_array' to numpy.float64.");
        return NULL;
    }
    // printf("Time array size: %d\n", (int)PyArray_SIZE(time_array_cast));
    //print_array_max(time_array_cast);
    int time_size = (int)PyArray_SIZE(time_array_cast);
    double *time = (double *)PyArray_DATA(time_array_cast);
    double *flux = (double *)PyArray_DATA(flux_array_cast);

    // Create edges array
    //printf("%f %f %d",  time[time_size - 1] , time[0], time_size);
    int binned_size =  (int) ceil((time[time_size - 1] - time[0]) / bin_size) + 1;
    double *edges = (double *)malloc((binned_size + 1) * sizeof(double));
    linspace(time[0], time[time_size - 1], binned_size + 1, edges);
    // printf("Binned size: %d\n", binned_size);

     // Allocate intermediate arrays
    double *binned_flux = (double *)malloc(binned_size * sizeof(double));
    double *binned_flux_err = (double *)malloc(binned_size * sizeof(double));
    int *count = (int *)malloc(binned_size * sizeof(int));
    double *mid_times = (double *)malloc(binned_size * sizeof(double));

    bin_data_fast(time, flux, time_size, edges, binned_flux, binned_flux_err, binned_size, count);

    // Count non-zero bins
    int non_zero_bins = 0;
    for (int i = 0; i < binned_size; i++) {
        if (count[i] > 0) non_zero_bins++;
    }

    // Allocate output arrays
    npy_intp dims[1] = {non_zero_bins};
    PyArrayObject *binned_flux_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject *binned_flux_err_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject *count_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_INT, 0);
    PyArrayObject *mid_times_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    double *filtered_flux = (double *)PyArray_DATA(binned_flux_array);
    double *filtered_flux_err = (double *)PyArray_DATA(binned_flux_err_array);
    int *filtered_count = (int *)PyArray_DATA(count_array);
    double *filtered_mid_times = (double *)PyArray_DATA(mid_times_array);

    // Populate filtered arrays
    int idx = 0;
    int warning = 0;
    for (int i = 0; i < binned_size; i++) 
    {
        if (count[i] > 0) {
            filtered_flux[idx] = binned_flux[i];
            filtered_flux_err[idx] = binned_flux_err[i];
            filtered_count[idx] = count[i];
            filtered_mid_times[idx] = (edges[i] + edges[i+1])/2; //mid_times[i];
            idx++;
        }
        if (count[i] == N_BIN_MAX)   warning=1;
    }
    if (warning == 1){
        printf("Warning: %d data points in a bin, data not binned properly!\n", N_BIN_MAX);
    }

    // Free temporary arrays
    free(edges);
    free(binned_flux);
    free(binned_flux_err);
    free(count);
    free(mid_times);

    // Return result as a tuple
    PyObject *result = Py_BuildValue("NNNN",
                                     PyArray_Return(mid_times_array),
                                     PyArray_Return(binned_flux_array),
                                     PyArray_Return(binned_flux_err_array),
                                     PyArray_Return(count_array));
    return result;
}













static PyObject *median_filter(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    double window;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOd", &x_obj, &y_obj, &window)) {
        return NULL;
    }

    // Ensure NumPy is initialized
    import_array();

    // Cast input arrays to Numpy arrays
    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROMANY(
        x_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (x_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'x' to numpy.float64 array.");
        return NULL;
    }

    PyArrayObject *y_array = (PyArrayObject *)PyArray_FROMANY(
        y_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (y_array == NULL) {
        Py_DECREF(x_array);
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'y' to numpy.float64 array.");
        return NULL;
    }

    // Ensure x and y have the same size
    int signal_length = (int)PyArray_SIZE(x_array);
    if (PyArray_SIZE(y_array) != signal_length) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "Input arrays 'x' and 'y' must have the same length.");
        return NULL;
    }

    // Allocate output array
    npy_intp dims[1] = {signal_length};
    PyArrayObject *output_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (output_array == NULL) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output array.");
        return NULL;
    }

    // Get pointers to data
    const double *x = (const double *)PyArray_DATA(x_array);
    const double *y = (const double *)PyArray_DATA(y_array);
    double *output_signal = (double *)PyArray_DATA(output_array);

    // Call the median filter
    median_filter_fast(x, y, output_signal, window, signal_length);

    // Clean up
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    // Return the output array
    return (PyObject *)output_array;
}


static PyObject *convolve_1d(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    double window;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOd", &x_obj, &y_obj, &window)) {
        return NULL;
    }

    // Ensure NumPy is initialized
    import_array();

    // Cast input arrays to Numpy arrays
    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROMANY(
        x_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (x_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'x' to numpy.float64 array.");
        return NULL;
    }

    PyArrayObject *y_array = (PyArrayObject *)PyArray_FROMANY(
        y_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (y_array == NULL) {
        Py_DECREF(x_array);
        PyErr_SetString(PyExc_TypeError, "Failed to cast 'y' to numpy.float64 array.");
        return NULL;
    }

    // Ensure x and y have the same size
    int signal_length = (int)PyArray_SIZE(x_array);
    if (PyArray_SIZE(y_array) != signal_length) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_ValueError, "Input arrays 'x' and 'y' must have the same length.");
        return NULL;
    }

    // Allocate output array
    npy_intp dims[1] = {signal_length};
    PyArrayObject *output_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (output_array == NULL) {
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output array.");
        return NULL;
    }

    // Get pointers to data
    const double *x = (const double *)PyArray_DATA(x_array);
    const double *y = (const double *)PyArray_DATA(y_array);
    double *output_signal = (double *)PyArray_DATA(output_array);

    // Call the median filter
    convolve_1d_fast(x, y, output_signal, window, signal_length);

    // Clean up
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    // Return the output array
    return (PyObject *)output_array;
}











static PyObject *rv1(PyObject *self, PyObject *args) {
    PyObject *a_g_obj;
    double t_zero, period, K1, e, w, incl, V0;
    int accurate_tp;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "Odddddddi", &a_g_obj, &t_zero, &period, &K1, &e, &w, &incl, &V0, &accurate_tp)) {
        return NULL;
    }

    // Initialize NumPy C API
    import_array();

    // Cast the input array to Numpy and ensure it is of type double
    PyArrayObject *a_g_array = (PyArrayObject *)PyArray_FROMANY(
        a_g_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (a_g_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Input array 'a_g' must be a 1D numpy array of type numpy.float64.");
        return NULL;
    }

    // Get the size of the input array
    int a_g_size = (int)PyArray_SIZE(a_g_array);

    // Allocate output array
    npy_intp dims[1] = {a_g_size};
    PyArrayObject *res_g_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (res_g_array == NULL) {
        Py_DECREF(a_g_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output array.");
        return NULL;
    }

    // Get pointers to the data
    const double *a_g = (const double *)PyArray_DATA(a_g_array);
    double *res_g = (double *)PyArray_DATA(res_g_array);

    // Call the rv1 function
    rv1_c(a_g, res_g, a_g_size, t_zero, period, K1, e, w, incl, V0, accurate_tp);

    // Decrease reference to input array
    Py_DECREF(a_g_array);

    // Return the output array
    return (PyObject *)res_g_array;
}




static PyObject *rv1_loglike(PyObject *self, PyObject *args) {
    PyObject *a_g_obj, *rv_obj, *rv_err_obj;
    double t_zero, period, K1, e, w, incl, V0, jitter;
    int accurate_tp, offset;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOOdddddddiid", &a_g_obj, &rv_obj, &rv_err_obj,
                          &t_zero, &period, &K1, &e, &w, &incl, &V0, &accurate_tp, &offset, &jitter)) {
        return NULL;
    }

    // Initialize NumPy C API
    import_array();

    // Convert input arrays to NPY_DOUBLE if not already of that type
    PyArrayObject *a_g_array = (PyArrayObject *)PyArray_FROMANY(a_g_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *rv_array = (PyArrayObject *)PyArray_FROMANY(rv_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *rv_err_array = (PyArrayObject *)PyArray_FROMANY(rv_err_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);

    if (!a_g_array || !rv_array || !rv_err_array) {
        PyErr_SetString(PyExc_TypeError, "All input arrays must be 1D numpy arrays of type numpy.float64.");
        Py_XDECREF(a_g_array);
        Py_XDECREF(rv_array);
        Py_XDECREF(rv_err_array);
        return NULL;
    }

    // Ensure all input arrays have the same size
    npy_intp size = PyArray_SIZE(a_g_array);
    if (PyArray_SIZE(rv_array) != size || PyArray_SIZE(rv_err_array) != size) {
        PyErr_SetString(PyExc_ValueError, "All input arrays must have the same size.");
        Py_DECREF(a_g_array);
        Py_DECREF(rv_array);
        Py_DECREF(rv_err_array);
        return NULL;
    }

    // Get pointers to the data in the arrays
    const double *a_g = (const double *)PyArray_DATA(a_g_array);
    const double *rv = (const double *)PyArray_DATA(rv_array);
    const double *rv_err = (const double *)PyArray_DATA(rv_err_array);

    // Initialize log-likelihood
    double loglike_val = 0.0;

    // Compute the log-likelihood in parallel using OpenMP
    #pragma omp parallel for reduction(-:loglike_val)
    for (npy_intp i = 0; i < size; i++) {
        // Compute the model RV for the given input
        double model = __rv1(a_g[i], t_zero, period, K1, e, w, incl, V0, accurate_tp);
        // Accumulate the log-likelihood using the provided jitter and offset
        loglike_val -= bruce_loglike(rv[i], rv_err[i], model, jitter, offset);
    }

    // Clean up
    Py_DECREF(a_g_array);
    Py_DECREF(rv_array);
    Py_DECREF(rv_err_array);

    // Return the log-likelihood as a Python float
    return PyFloat_FromDouble(loglike_val);
}



static PyObject *rv2(PyObject *self, PyObject *args) {
    PyObject *a_g_obj;
    double t_zero, period, K1, K2, e, w, incl, V0;
    int accurate_tp;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "Oddddddddi", &a_g_obj, &t_zero, &period, &K1, &K2, &e, &w, &incl, &V0, &accurate_tp)) {
        return NULL;
    }

    // Initialize NumPy C API
    import_array();

    // Cast the input array to Numpy and ensure it is of type double
    PyArrayObject *a_g_array = (PyArrayObject *)PyArray_FROMANY(
        a_g_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (a_g_array == NULL) {
        PyErr_SetString(PyExc_TypeError, "Input array 'a_g' must be a 1D numpy array of type numpy.float64.");
        return NULL;
    }

    // Get the size of the input array
    int a_g_size = (int)PyArray_SIZE(a_g_array);

    // Allocate output arrays
    npy_intp dims[1] = {a_g_size};
    PyArrayObject *res_g1_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyArrayObject *res_g2_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (res_g1_array == NULL || res_g2_array == NULL) {
        Py_DECREF(a_g_array);
        Py_XDECREF(res_g1_array);
        Py_XDECREF(res_g2_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the output arrays.");
        return NULL;
    }

    // Get pointers to the data
    const double *a_g = (const double *)PyArray_DATA(a_g_array);
    double *res_g1 = (double *)PyArray_DATA(res_g1_array);
    double *res_g2 = (double *)PyArray_DATA(res_g2_array);

    // Call the rv2_c function
    rv2_c(a_g, res_g1, res_g2, a_g_size, t_zero, period, K1, K2, e, w, incl, V0, accurate_tp);

    // Decrease reference to input array
    Py_DECREF(a_g_array);

    // Return the output arrays as a tuple
    return Py_BuildValue("NN", res_g1_array, res_g2_array);
}

static PyObject *rv2_loglike(PyObject *self, PyObject *args) {
    PyObject *a_g_obj, *rv1_obj, *rv2_obj, *rv1_err_obj, *rv2_err_obj;
    double t_zero, period, K1, K2, e, w, incl, V0, jitter;
    int accurate_tp, offset;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOOOOddddddddiid", &a_g_obj, &rv1_obj, &rv2_obj, 
                          &rv1_err_obj, &rv2_err_obj, &t_zero, &period, &K1, &K2, 
                          &e, &w, &incl, &V0, &accurate_tp, &offset, &jitter)) {
        return NULL;
    }

    // Initialize NumPy C API
    import_array();

    // Convert input arrays to NPY_DOUBLE if not already of that type
    PyArrayObject *a_g_array = (PyArrayObject *)PyArray_FROMANY(a_g_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *rv1_array = (PyArrayObject *)PyArray_FROMANY(rv1_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *rv2_array = (PyArrayObject *)PyArray_FROMANY(rv2_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *rv1_err_array = (PyArrayObject *)PyArray_FROMANY(rv1_err_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *rv2_err_array = (PyArrayObject *)PyArray_FROMANY(rv2_err_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);

    if (!a_g_array || !rv1_array || !rv2_array || !rv1_err_array || !rv2_err_array) {
        PyErr_SetString(PyExc_TypeError, "All input arrays must be 1D numpy arrays of type numpy.float64.");
        Py_XDECREF(a_g_array);
        Py_XDECREF(rv1_array);
        Py_XDECREF(rv2_array);
        Py_XDECREF(rv1_err_array);
        Py_XDECREF(rv2_err_array);
        return NULL;
    }

    // Ensure all input arrays have the same size
    npy_intp size = PyArray_SIZE(a_g_array);
    if (PyArray_SIZE(rv1_array) != size || PyArray_SIZE(rv2_array) != size ||
        PyArray_SIZE(rv1_err_array) != size || PyArray_SIZE(rv2_err_array) != size) {
        PyErr_SetString(PyExc_ValueError, "All input arrays must have the same size.");
        Py_DECREF(a_g_array);
        Py_DECREF(rv1_array);
        Py_DECREF(rv2_array);
        Py_DECREF(rv1_err_array);
        Py_DECREF(rv2_err_array);
        return NULL;
    }

    // Get pointers to the data in the arrays
    const double *a_g = (const double *)PyArray_DATA(a_g_array);
    const double *rv1 = (const double *)PyArray_DATA(rv1_array);
    const double *rv2 = (const double *)PyArray_DATA(rv2_array);
    const double *rv1_err = (const double *)PyArray_DATA(rv1_err_array);
    const double *rv2_err = (const double *)PyArray_DATA(rv2_err_array);

    // Initialize log-likelihood
    double loglike_val = 0.0;

    // Compute the log-likelihood in parallel using OpenMP
    #pragma omp parallel for reduction(-:loglike_val)
    for (npy_intp i = 0; i < size; i++) {
        // Compute the model RV components for the given input
        double model1, model2;
        rv2_c(&a_g[i], &model1, &model2, 1, t_zero, period, K1, K2, e, w, incl, V0, accurate_tp);

        // Accumulate the log-likelihood for both RV1 and RV2
        loglike_val -= bruce_loglike(rv1[i], rv1_err[i], model1, jitter, offset);
        loglike_val -= bruce_loglike(rv2[i], rv2_err[i], model2, jitter, offset);
    }

    // Clean up
    Py_DECREF(a_g_array);
    Py_DECREF(rv1_array);
    Py_DECREF(rv2_array);
    Py_DECREF(rv1_err_array);
    Py_DECREF(rv2_err_array);

    // Return the log-likelihood as a Python float
    return PyFloat_FromDouble(loglike_val);
}






static PyObject *check_proximity_of_timestamps(PyObject *self, PyObject *args) {
    PyObject * time_trial_obj, *time_obj;
    double width;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOd", &time_trial_obj, &time_obj,  &width)) {
        return NULL;
    }

    // Initialize NumPy C API
    import_array();

    PyArrayObject *time_trial_array = (PyArrayObject *)PyArray_FROMANY(time_trial_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *time_array = (PyArrayObject *)PyArray_FROMANY(time_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);

    if (!time_trial_array || !time_array) {
        PyErr_SetString(PyExc_TypeError, "All input arrays must be 1D numpy arrays of type numpy.float64.");
        Py_XDECREF(time_trial_array);
        Py_XDECREF(time_array);
        return NULL;
    }

    // Get the size and allocate the mask array
    int time_trial_size = (int)PyArray_SIZE(time_trial_array);
    int time_size = (int)PyArray_SIZE(time_array);
    npy_intp dims[1] = {time_trial_size};
    PyArrayObject *mask_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_BOOL);

    // Get pointers to the data in the arrays
    const double *time_trial = (const double *)PyArray_DATA(time_trial_array);
    const double *time = (const double *)PyArray_DATA(time_array);
    _Bool *mask = (_Bool *)PyArray_DATA(mask_array);

    check_proximity_of_timestamps_fast(time_trial, time, time_trial_size,time_size, width, mask);
    // Decrease reference to input array
    Py_DECREF(time_trial_array);
    Py_DECREF(time_array);
    return (PyObject *)mask_array;
}





static PyObject *template_match_reduce(PyObject *self, PyObject *args) 
{
    // Allocation
    PyObject * time_trial_obj, *time_obj, *flux_obj, *flux_err_obj, *normalisation_model_obj;
    double width, period, radius_1, k, incl, e, w, c, alpha, cadence, light_3, jitter;
    int noversample, ld_law, accurate_tp, offset;
    // printf("I got here");
    // fflush(stdout);
    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOOOOddddddddddidiidi", &time_trial_obj,
                                                        &time_obj, &flux_obj, &flux_err_obj, &normalisation_model_obj,
                                                        &width,
                                                        &period,
                                                        &radius_1, &k, &incl,
                                                        &e, &w,
                                                        &c, &alpha,
                                                        &cadence, &noversample,
                                                        &light_3, 
                                                        &ld_law, 
                                                        &accurate_tp,
                                                        &jitter, &offset )) {
        return NULL;
    }

    // printf("I parsed");
    // fflush(stdout);
    // Convert input arrays to NPY_DOUBLE if not already of that type
    PyArrayObject *time_trial_array = (PyArrayObject *)PyArray_FROMANY(time_trial_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *time_array = (PyArrayObject *)PyArray_FROMANY(time_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *flux_array = (PyArrayObject *)PyArray_FROMANY(flux_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *flux_err_array = (PyArrayObject *)PyArray_FROMANY(flux_err_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *normalisation_model_array = (PyArrayObject *)PyArray_FROMANY(normalisation_model_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    if (!time_trial_array || !time_array || !flux_array || !flux_err_array || !normalisation_model_array) {
        PyErr_SetString(PyExc_TypeError, "All input arrays must be 1D numpy arrays of type numpy.float64.");
        Py_XDECREF(time_trial_array);
        Py_XDECREF(time_array);
        Py_XDECREF(flux_array);
        Py_XDECREF(flux_err_array);
        Py_XDECREF(normalisation_model_array);
        return NULL;
    }
    // printf("I formatted");
    // fflush(stdout);

    // Lets get the sizes
    int time_trial_size = (int)PyArray_SIZE(time_trial_array);
    int time_size = (int)PyArray_SIZE(time_array);

    // Now lets allocate DeltaL
    npy_intp dims[1] = {time_trial_size};
    PyArrayObject *DeltaL_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    // printf("I got the sizes");
    // fflush(stdout);

    // Lets get the double pointers
    const double *time_trial = (const double *)PyArray_DATA(time_trial_array);
    double *DeltaL = (double *)PyArray_DATA(DeltaL_array);
    const double *time = (const double *)PyArray_DATA(time_array);
    const double *flux = (const double *)PyArray_DATA(flux_array);
    const double *flux_err = (const double *)PyArray_DATA(flux_err_array);
    const double *normalisation_model = (const double *)PyArray_DATA(normalisation_model_array);

    // printf("I converted to douvle *");
    // fflush(stdout);

    // Now lets make the call
    template_match(
        time_trial, DeltaL,
        time, flux, flux_err, normalisation_model, 
        time_trial_size, time_size,
        width,
        period,
        radius_1, k, incl,
        e, w,
        c, alpha,
        cadence, noversample,
        light_3,
        ld_law,
        accurate_tp,
        jitter, offset);
    // printf("I made the template match call");
    // fflush(stdout);

    // Decrease the references
    Py_DECREF(time_trial_array);
    Py_DECREF(time_array);
    Py_DECREF(flux_array);
    Py_DECREF(flux_err_array);
    Py_DECREF(normalisation_model_array);
    // printf("I dec reffed");
    // fflush(stdout);

    // Return the array
    return (PyObject *) DeltaL_array;
}











static PyObject *template_match_batch_reduce(PyObject *self, PyObject *args) 
{
    // Allocation
    PyObject * time_trial_obj, *time_obj, *flux_obj, *flux_err_obj, *normalisation_model_obj, *radius_1_obj, *k_obj, *incl_obj;
    double period, e, w, c, alpha, cadence, light_3, jitter;
    int noversample, ld_law, accurate_tp, offset;
    // printf("I got here");
    // fflush(stdout);
    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOOOOdOOOdddddidiidi", &time_trial_obj,
                                                        &time_obj, &flux_obj, &flux_err_obj, &normalisation_model_obj,
                                                        &period,
                                                        &radius_1_obj, &k_obj, &incl_obj,
                                                        &e, &w,
                                                        &c, &alpha,
                                                        &cadence, &noversample,
                                                        &light_3, 
                                                        &ld_law, 
                                                        &accurate_tp,
                                                        &jitter, &offset )) {
        return NULL;
    }

    // printf("I parsed");
    // fflush(stdout);
    // Convert input arrays to NPY_DOUBLE if not already of that type
    PyArrayObject *time_trial_array = (PyArrayObject *)PyArray_FROMANY(time_trial_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *time_array = (PyArrayObject *)PyArray_FROMANY(time_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *flux_array = (PyArrayObject *)PyArray_FROMANY(flux_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *flux_err_array = (PyArrayObject *)PyArray_FROMANY(flux_err_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *normalisation_model_array = (PyArrayObject *)PyArray_FROMANY(normalisation_model_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *radius_1_array = (PyArrayObject *)PyArray_FROMANY(radius_1_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *k_array = (PyArrayObject *)PyArray_FROMANY(k_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *incl_array = (PyArrayObject *)PyArray_FROMANY(incl_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);


    if (!time_trial_array || !time_array || !flux_array || !flux_err_array || !normalisation_model_array || !radius_1_array || !k_array || !incl_array) {
        PyErr_SetString(PyExc_TypeError, "All input arrays must be 1D numpy arrays of type numpy.float64.");
        Py_XDECREF(time_trial_array);
        Py_XDECREF(time_array);
        Py_XDECREF(flux_array);
        Py_XDECREF(flux_err_array);
        Py_XDECREF(normalisation_model_array);
        Py_XDECREF(radius_1_array);
        Py_XDECREF(k_array);
        Py_XDECREF(incl_array);
        return NULL;
    }
    // printf("I formatted");
    // fflush(stdout);

    // Lets get the sizes
    int time_trial_size = (int)PyArray_SIZE(time_trial_array);
    int time_size = (int)PyArray_SIZE(time_array);
    int radius_1_size = (int)PyArray_SIZE(radius_1_array);
    int k_size = (int)PyArray_SIZE(k_array);
    int incl_size = (int)PyArray_SIZE(incl_array);

    // Now lets allocate DeltaL
    npy_intp dims[1] = {time_trial_size*radius_1_size*k_size*incl_size};
    PyArrayObject *DeltaL_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    // printf("I got the sizes");
    // fflush(stdout);

    // Lets get the double pointers
    const double *time_trial = (const double *)PyArray_DATA(time_trial_array);
    double *DeltaL = (double *)PyArray_DATA(DeltaL_array);
    const double *time = (const double *)PyArray_DATA(time_array);
    const double *flux = (const double *)PyArray_DATA(flux_array);
    const double *flux_err = (const double *)PyArray_DATA(flux_err_array);
    const double *normalisation_model = (const double *)PyArray_DATA(normalisation_model_array);
    const double *radius_1 = (const double *)PyArray_DATA(radius_1_array);
    const double *k = (const double *)PyArray_DATA(k_array);
    const double *incl = (const double *)PyArray_DATA(incl_array);

    // printf("I converted to douvle *");
    // fflush(stdout);

    // Now lets make the call
    template_match_batch(
        time_trial, DeltaL,
        time, flux, flux_err, normalisation_model, 
        time_trial_size, time_size,
        period,
        radius_1, k, incl,
        radius_1_size,k_size,incl_size,
        e, w,
        c, alpha,
        cadence, noversample,
        light_3,
        ld_law,
        accurate_tp,
        jitter, offset);

    // create the 4d array
    npy_intp dims4d[4] = {time_trial_size, radius_1_size, k_size, incl_size};
    PyArray_Dims new_dims = {dims4d, 4};
    PyObject* reshaped_array = PyArray_Newshape((PyArrayObject*) DeltaL_array, &new_dims, NPY_ANYORDER);

    // Decrease the references
    Py_DECREF(time_trial_array);
    Py_DECREF(time_array);
    Py_DECREF(flux_array);
    Py_DECREF(flux_err_array);
    Py_DECREF(normalisation_model_array);
    Py_DECREF(radius_1_array);
    Py_DECREF(k_array);
    Py_DECREF(incl_array);
    Py_DECREF(DeltaL_array);

    // Return the array
    return (PyObject *) reshaped_array;
}


















static PyObject *phase_dispersion(PyObject *self, PyObject *args) 
{
    // Allocation
    PyObject * time_trial_obj, * peaks_obj,  *periods_obj, *time_obj, *flux_obj, *flux_err_obj;

    // Parse input arguments
    if (!PyArg_ParseTuple(args, "OOOOOO", &time_trial_obj,&peaks_obj,&periods_obj, &time_obj, &flux_obj, &flux_err_obj)) {
        return NULL;
    }

    // re-cast the arrays
    PyArrayObject *time_trial_array = (PyArrayObject *)PyArray_FROMANY(time_trial_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *peaks_array = (PyArrayObject *)PyArray_FROMANY(peaks_obj, NPY_INT, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *periods_array = (PyArrayObject *)PyArray_FROMANY(periods_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *time_array = (PyArrayObject *)PyArray_FROMANY(time_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *flux_array = (PyArrayObject *)PyArray_FROMANY(flux_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *flux_err_array = (PyArrayObject *)PyArray_FROMANY(flux_err_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS);

    // Lets get the sizes
    int peaks_size = (int)PyArray_SIZE(peaks_array);
    int periods_size = (int)PyArray_SIZE(periods_array);
    int time_size = (int)PyArray_SIZE(time_array);

    // Now lets allocate the dispersion array (zeros)
    npy_intp dims[1] = {periods_size};
    PyArrayObject *dispersion_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    PyArrayObject *L_array = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);

    // Now cast to doubles
    const double *time_trial = (const double *)PyArray_DATA(time_trial_array);
    const int *peaks = (const int *)PyArray_DATA(peaks_array);
    const double *periods = (const double *)PyArray_DATA(periods_array);
    const double *time = (const double *)PyArray_DATA(time_array);
    const double *flux = (const double *)PyArray_DATA(flux_array);
    const double *flux_err = (const double *)PyArray_DATA(flux_err_array);
    double *dispersion = (double *) PyArray_DATA(dispersion_array);
    double *L = (double *) PyArray_DATA(L_array);

    // Now make the call
    compute_dispersion(time_trial,
        peaks,
        periods,
        time, flux, flux_err,
        dispersion, L,
        peaks_size,
        periods_size, time_size);


    // Decrease the references
    Py_DECREF(time_trial_array);
    Py_DECREF(peaks_array);
    Py_DECREF(periods_array);
    Py_DECREF(time_array);
    Py_DECREF(flux_array);
    Py_DECREF(flux_err_array);


   // Return result as a tuple
    PyObject *result = Py_BuildValue("NN",
                                     PyArray_Return(dispersion_array),
                                     PyArray_Return(L_array));
    return result;
}





























// Define the methods in the module
static PyMethodDef bruce_c_methods[] = {
    {"loglike", loglike, METH_VARARGS, 
    "Calculate the log-likliehood of the data given the model, errors, jitter, and offset.\n"
    "Assumes y,yerr, and model are arrays of same shape. \n"
    "\n"
    "Parameters:\n"
    "    y: array of data points                              (np.float64)\n"
    "    yerr: array of errors                                (np.float64)\n"
    "    model: array of model values                         (np.float64)\n"
    "    jitter: Jitter Value added in quadrature to yerr.    (float) \n"
    "    offset: Subtract an arbritrary offset when set to 1. (int)\n"
    "\n"
    "Returns:\n"
    "    log-likelihood                                       (np.float64)"},

    {"lc", lc, METH_VARARGS, "Compute the light curve for an array of times using the given parameters."},
    {"lc_loglike", lc_loglike, METH_VARARGS, "Compute the light curve log-likliehood for an array of times using the given parameters."},
    {"bin_data", bin_data, METH_VARARGS, "Bin data with optional linspace for edges."},
    {"median_filter", median_filter, METH_VARARGS, "median filter data given a filter width."},
    {"convolve_1d", convolve_1d, METH_VARARGS, "Convolve filter data given a filter width."},   
    {"rv1", rv1, METH_VARARGS, "The radial velocity of star 1 given a set of parameters."}, 
    {"rv1_loglike", rv1_loglike, METH_VARARGS, "Compute the radial velocity log-likliehood given an RV dataset."}, 
    {"rv2", rv2, METH_VARARGS, "The radial velocity of star 1 and star2 given a set of parameters."},  
    {"rv2_loglike", rv2_loglike, METH_VARARGS, "Compute the radial velocity log-likliehood given an RV dataset."}, 
    {"check_proximity_of_timestamps", check_proximity_of_timestamps, METH_VARARGS, "Keep trial epochs with any observation within +/- width (full width, not 0.5*width)."},
    {"template_match_reduce", template_match_reduce, METH_VARARGS, "Template match a lighcurve."},
    {"phase_dispersion", phase_dispersion, METH_VARARGS, "Calculate the dispersion."},
    {"template_match_batch_reduce", template_match_batch_reduce, METH_VARARGS, "Template match batch [radius_1, k, incl]."},
    {NULL, NULL, 0, NULL}  // Sentinel
};



// Define the module
static struct PyModuleDef bruce_c_module = {
    PyModuleDef_HEAD_INIT,
    "bruce_c",  // Module name
    NULL,        // Module documentation
    -1,          // Module keeps state in global variables
    bruce_c_methods
};

// Initialize the module
PyMODINIT_FUNC PyInit_bruce_c(void) {
    import_array();  // Initialize the NumPy C API
    return PyModule_Create(&bruce_c_module);
}
