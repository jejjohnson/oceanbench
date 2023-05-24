import sys
from itertools import product

import intake

def all_params():
    
    all_params = {}

    cat = intake.open_catalog('./catalog.yaml')
    for item in cat:
        description = cat[item].describe()
        params = description["user_parameters"]
        params = {params[i]["name"]: params[i]["allowed"] for i in range(len(params))}

        # clean-up blank values if needed
        seasons = [s for s in params["season"] if s != ""]
        if "grid" in params.keys():
            grids = [g for g in params["grid"] if g != ""]

        # FESOM is currently the only item without a "region" parameter
        if "region" not in params.keys():
            cat_kwargs = [p for p in product(params["datatype"], seasons)]
            cat_kwargs = [{"datatype": i[0], "season": i[1]} for i in cat_kwargs]
        else:
            non_grid_datatypes = [d for d in params["datatype"] if d != "grid"]
            cat_kwargs = [
                p for p in product(params["region"], non_grid_datatypes, seasons)
            ]
            cat_kwargs = [{"region": i[0], "datatype": i[1], "season": i[2]} for i in cat_kwargs]

            if "grid" in params.keys():
                more_kwargs = [p for p in product(params["region"], ["grid"], grids)]
                more_kwargs = [
                    {"region": i[0], "datatype": i[1], "grid": i[2]} for i in more_kwargs
                ]
                cat_kwargs = cat_kwargs + more_kwargs

        all_params.update({item: cat_kwargs})

    return all_params, cat


def main(params_only=False, all_params=all_params):

    all_params, cat = all_params()

    for item in all_params.keys():
        print(f"\n{item}")
        print(f"{len(all_params[item])} parameterizations for {item}: {all_params[item]}")

        if not params_only:
            for d in all_params[item]:
                print(f"\n\n{item}: loading parameterization {d}")
                # this specific dataset has not been added yet, and I don't think there's a way
                # to skip it in the `catalog.yaml` user paramaters, so we skip it manually here:
                if item == "FESOM" and d["datatype"] == "int" and d["season"] == "aso":
                    pass
                else:
                    ds = cat[item](**d).to_dask()
                    print(ds)


if __name__ == "__main__":
    if "params_only" in sys.argv:
        main(params_only=True)
    else:
        main()
