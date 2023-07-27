| Variant                           | CuPy                  | CrossPy               | Parla                 | Parray                | Working?                  |
| --------------------------------- | --------------------- | --------------------- | --------------------- | --------------------- | ------------------------- |
| quicksort_cupy                    | :heavy_check_mark:    |                       |                       |                       | :white_check_mark:        |
| quicksort_crosspy                 |                       | :heavy_check_mark:    |                       |                       | :white_check_mark:        |
| quicksort_parla_cupy              | :heavy_check_mark:    |                       | :heavy_check_mark:    |                       | :white_check_mark:        |
| quicksort_parla_crosspy           |                       | :heavy_check_mark:    | :heavy_check_mark:    |                       | :white_check_mark:        |
| quicksort_parla_cupy_parray       | :heavy_check_mark:    |                       | :heavy_check_mark:    | :heavy_check_mark:    | :x:                       |
| quicksort_parla_crosspy_parray    |                       | :heavy_check_mark:    | :heavy_check_mark:    | :heavy_check_mark:    | :x:                       |

```
python example/quicksort/quicksort_parla_crosspy.py -depth 10 -m 200000000 -num_gpus 1 --no-verify
```