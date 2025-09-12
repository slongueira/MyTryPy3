import os
import logging
import warnings
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import peak_widths
from scipy.integrate import simpson as simps
from openpyxl.utils import get_column_letter
from MyLoadData import LoadFiles


# %% --------------------------------------------------------------------------
# INITIAL CONFIGURATION
# -----------------------------------------------------------------------------

logging.basicConfig(level=logging.ERROR,
                    format="%(levelname)s: %(message)s"
)

REQ_LEVEL = 25
logging.addLevelName(REQ_LEVEL, "REQ")

def req(self, message, *args, **kwargs):
    if self.isEnabledFor(REQ_LEVEL):
        self._log(REQ_LEVEL, message, args, **kwargs)

logging.Logger.req = req
logger1 = logging.getLogger(__name__)

if __name__ == "__main__":
    logger1.setLevel(logging.INFO)


mpl.use("Qt5Agg")
plt.close("all")
plt.ion()


# %% --------------------------------------------------------------------------
# PATHS DEFINITION AND DATA INITIALIZATION
# -----------------------------------------------------------------------------

def select_paths():
    """
    Opens file dialogs to select directories and configuration files.
    
    Returns
    -------
    df_exps : pd.DataFrame or None
        Experiment metadata with merged load info.
    reports_dir : str or None
        Path to the directory where PDF reports are stored.
    datasets_dir : str or None
        Path to the directory where processed datasets are stored.
    """
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    
    # Ask for experiments directory
    logger1.req("Please provide the experimets directory location.")
    exps_dir = filedialog.askdirectory(title="Select Experiments Directory")
    if exps_dir:
        exps_dir = os.path.normpath(exps_dir)
    else:
        logger1.info("Directory selection canceled.")
        return None, None, None
    
    # Define RawData folder
    rawdata_dir = os.path.join(exps_dir, "RawData")
    if not os.path.isdir(rawdata_dir):
        logger1.error("Could not find RawData folder in experiments directory.")
        return None, None, None
    
    # Define/Create Reports folder
    reports_dir = os.path.join(exps_dir, "Reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Define/Create DataSets folder
    datasets_dir = os.path.join(exps_dir, "DataSets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Define ExpsDescription file 
    exps_file = os.path.join(exps_dir, "ExpsDescription.xlsx")
    if not os.path.isfile(exps_file):
        logger1.error("Could not find ExpsDescription.xlsx file in experiments directory.")
        return None, None, None
    
    # Define LoadsDescription file
    loads_file = os.path.join(exps_dir, "LoadsDescription.xlsx")
    if not os.path.isfile(loads_file):
        logger1.error("Could not find LoadsDescription.xlsx file in experiments directory.")
        return None, None, None
    
    # Load ExpsDescription and LoadDescription files
    df_exps = pd.read_excel(exps_file)
    df_loads = pd.read_excel(loads_file)
    
    # Ask for TribuId value
    logger1.req("Please provide a TribuId value to analyze.")
    tribu_id = simpledialog.askstring(title="TribuId Selection", prompt="Enter a valid TribuId value:")
    if not tribu_id:
        logger1.info("TribuId selection canceled.")
        return None, None, None
    elif tribu_id in df_exps['TribuId'].dropna().unique():
        logger1.info(f"TribuId value correctly saved: {tribu_id}")
        df_exps = df_exps[df_exps['TribuId'] == tribu_id]
    else:
        logger1.error(f"Invalid TribuId value: {tribu_id}")
        return None, None, None
    
    # Merge with load info
    loads_fields = ("Req", "Gain", "Ceq")
    for field in loads_fields:
        if field not in df_exps.columns:
            df_exps.insert(2, field, None)
    
    for idx, row in df_exps.iterrows():
        if row.RloadId in df_loads.RloadId.values:
            for field in loads_fields:
                df_exps.loc[idx, field] = df_loads.loc[df_loads.RloadId == row.RloadId, field].values[0]
        elif row.RloadId == "ElectrodeImpedance":
            logger1.warning(f"Load {row.RloadId} is Electrode Impedance. Assigned to 80 kOhms.")
            df_exps.loc[idx, loads_fields] = [80e3, 1, None]
        else:
            logger1.warning(f"Load {row.RloadId} not found. Assigned open circuit.")
            df_exps.loc[idx, loads_fields] = [float("inf"), 1, None]
        
        # Check files existence
        daq_file = os.path.join(rawdata_dir, row.DaqFile)
        motor_file = os.path.join(rawdata_dir, row.MotorFile)
        
        if os.path.isfile(daq_file):
            df_exps.loc[idx, "DaqFile"] = daq_file
        else:
            logger1.warning(f"File {daq_file} not found. Experiment {row.ExpId} dropped.")
            df_exps.drop(idx, inplace=True)
            continue
        
        if os.path.isfile(motor_file):
            df_exps.loc[idx, "MotorFile"] = motor_file
        else:
            logger1.warning(f"File {motor_file} not found. Experiment {row.ExpId} dropped.")
            df_exps.drop(idx, inplace=True)
            continue
    
    return df_exps, reports_dir, datasets_dir


# %% --------------------------------------------------------------------------
# CYCLE ANALYSIS
# -----------------------------------------------------------------------------

def cycle_analysis(cycle, req_value):
    """Analyzes a single cycle and returns metrics as dict."""
    
    imax = cycle.Voltage.idxmax()
    imin = cycle.Voltage.idxmin()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pos_width = peak_widths(cycle.Voltage.values, [imax], rel_height=0.5)[0][0] * cycle.Time.diff().mean()
        neg_width = peak_widths(cycle.Voltage.values, [imin], rel_height=0.5)[0][0] * cycle.Time.diff().mean()
    
    cycle["Power"] = cycle["Voltage"]**2 / req_value
    
    cycle_pos = cycle[cycle["Voltage"] > 0].copy()
    cycle_neg = cycle[cycle["Voltage"] < 0].copy()
    cycle_neg["Voltage"] = -cycle_neg["Voltage"]
    
    return {"VoltageMax": cycle.loc[imax, "Voltage"],
            "VoltageMin": cycle.loc[imin, "Voltage"],
            "PosPeakWidth": pos_width,
            "NegPeakWidth": neg_width,
            "PosEnergy": simps(cycle_pos["Power"], x=cycle_pos["Time"]),
            "NegEnergy": simps(cycle_neg["Power"], x=cycle_neg["Time"]),
    }


# %% --------------------------------------------------------------------------
# EXPERIMENT ANALYSIS
# -----------------------------------------------------------------------------

# --- Plot Configuration ---
PLOTS_CONFIGURATION = {
    # Cycle range to plot (default: None)
    "cy_min": None,
    "cy_max": 10,

    # Signals to plot (comment out any signal you don't want to show)
    "Voltage": {"signal": "Voltage",
                "color": "gold",
                "linestyle": "-",
                "label": "Voltage (V)"
               },

    # "Current": {"signal": "Current",
    #             "color": "red",
    #             "linestyle": "-",
    #             "label": "Current (uA)"
    #            },

    # "State": {"signal": "State",
    #           "color": "blue",
    #           "linestyle": "-",
    #           "label": "State"
    #          },

    # "Position": {"signal": "Position",
    #              "color": "black",
    #              "linestyle": "--",
    #              "label": "Position (mm)"
    #             },

    # "Force": {"signal": "Force",
    #           "color": "orange",
    #           "linestyle": "--",
    #           "label": "Force (N)"
    #          },

    # "TargetForce": {"signal": "TargetForce",
    #                 "color": "brown",
    #                 "linestyle": "-.",
    #                 "label": "Target Force (N)"
    #                },

    # "Velocity": {"signal": "Velocity",
    #              "color": "magenta",
    #              "linestyle": "--",
    #              "label": "Velocity (m/s)"
    #             },

    # "Acceleration": {"signal": "Acceleration",
    #                  "color": "purple",
    #                  "linestyle": "-.",
    #                  "label": "Acceleration (m/s$^2$)"
    #                 }
}

# --- Experiment Analysis Function ---
def experiment_analysis(row, df_data, cycles_list, pdf):
    """Analyze a single experiment and append figures to a PDF."""

    exp_df = pd.DataFrame()

    # Create figures
    overlapped_cycles, ax_overlapped = plt.subplots()
    sequential_cycles, ax_sequential = plt.subplots()
    first = True

    # Iterate over cycles
    for cy_idx, cycle in cycles_list:

        # Determine cycle range to plot
        cy_min = PLOTS_CONFIGURATION["cy_min"] or 0
        cy_max = PLOTS_CONFIGURATION["cy_max"] or len(cycles_list) - 1

        if cy_min <= cy_idx <= cy_max:
            t_rel = cycle["Time"] - cycle["Time"].iloc[0]
            ax_sequential.axvline(x=cycle["Time"].iloc[-1], color="cyan", linestyle=":")
            
            # Plot each signal
            for signal in list(PLOTS_CONFIGURATION.values())[2:]:
                ax_overlapped.plot(
                    t_rel,
                    cycle[signal["signal"]],
                    color=signal["color"],
                    linestyle=signal["linestyle"],
                    label=signal["label"] if first else None
                )
                ax_sequential.plot(
                    cycle["Time"],
                    cycle[signal["signal"]],
                    color=signal["color"],
                    linestyle=signal["linestyle"],
                    label=signal["label"] if first else None
                )

            first = False

        # Compute metrics for cycles except the first one
        if cy_idx > 0:
            metrics = cycle_analysis(cycle, row.Req)
            metrics["cy_idx"] = cy_idx
            metrics["TotEnergy"] = metrics["PosEnergy"] + metrics["NegEnergy"]
            temp_df = pd.DataFrame([metrics])
            cols = ["cy_idx"] + [c for c in temp_df.columns if c != "cy_idx"]
            temp_df = temp_df[cols]
            exp_df = pd.concat([exp_df, temp_df], ignore_index=True)

    # Configure and save overlapped cycles figure
    ax_overlapped.set_xlabel("Time per cycle (s)")
    ax_overlapped.set_ylabel("Signals")
    ax_overlapped.set_title(f"Exp: {row.ExpId}, Req: {row.Req/1e6:.0f} M")
    ax_overlapped.grid(True)
    overlapped_cycles.legend()
    overlapped_cycles.tight_layout()
    pdf.savefig(overlapped_cycles)
    plt.close(overlapped_cycles)

    # Configure and save sequential cycles figure
    ax_sequential.set_xlabel("Time (s)")
    ax_sequential.set_ylabel("Signals")
    ax_sequential.set_title(f"Exp: {row.ExpId}, Req: {row.Req/1e6:.0f} M")
    ax_sequential.grid(True)
    sequential_cycles.legend()
    sequential_cycles.tight_layout()
    pdf.savefig(sequential_cycles)
    plt.close(sequential_cycles)

    return exp_df
    

# %% --------------------------------------------------------------------------
# SUMMARY PLOTS
# -----------------------------------------------------------------------------

def generate_summary_plots(summary_df, pdf):
    """Generates summary plots across exps and saves them into PDF."""
    
    req = summary_df["Req"]
    tribu_id = summary_df["TribuId"].unique()[0]
    
    # Power vs Req
    plt.figure()
    power = summary_df["AvgTotEnergy"] / summary_df["Duration"]
    power_err = np.sqrt(summary_df["VarTotEnergy"] / summary_df["Duration"])
    plt.plot(req, power, label="Power", color="green")
    plt.fill_between(req, power - power_err, power + power_err, color="green", alpha=0.3)
    plt.xlabel("Req (Ohm)")
    plt.ylabel("Power (W)")
    plt.title(f"TribuId: {tribu_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Energy vs Req
    plt.figure()
    for energy_type, color in [("Pos", "red"), ("Neg", "blue"), ("Tot", "green")]:
        avg = summary_df[f"Avg{energy_type}Energy"]
        err = np.sqrt(summary_df[f"Var{energy_type}Energy"])
        plt.plot(req, avg, label=f"{energy_type}Energy", color=color)
        plt.fill_between(req, avg - err, avg + err, color=color, alpha=0.3)
    plt.xlabel("Req (Ohm)")
    plt.ylabel("Energy (J)")
    plt.title(f"TribuId: {tribu_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Voltage vs Req
    plt.figure()
    for peak_type, color in [("Max", "red"), ("Min", "blue")]:
        avg = summary_df[f"AvgVoltage{peak_type}"]
        err = np.sqrt(summary_df[f"VarVoltage{peak_type}"])
        plt.plot(req, avg, label=f"Voltage{peak_type}", color=color)
        plt.fill_between(req, avg - err, avg + err, color=color, alpha=0.3)
    plt.xlabel("Req (Ohm)")
    plt.ylabel("Voltage (V)")
    plt.title(f"TribuId: {tribu_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Peak Widths vs Req
    plt.figure()
    for polarity, color in [("Pos", "red"), ("Neg", "blue")]:
        avg = summary_df[f"Avg{polarity}PeakWidth"]
        err = np.sqrt(summary_df[f"Var{polarity}PeakWidth"])
        plt.plot(req, avg, label=f"{polarity}PeakWidth", color=color)
        plt.fill_between(req, avg - err, avg + err, color=color, alpha=0.3)
    plt.xlabel("Req (Ohm)")
    plt.ylabel("Peak Widths (s)")
    plt.title(f"TribuId: {tribu_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


# %% --------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------

def main():
    df_exps, reports_dir, datasets_dir = select_paths()
    
    if df_exps is None:
        return
    
    plt.ioff()
    exps_summary = []
    tribu_id = df_exps.TribuId.unique()[0]
    
    pdf_path = os.path.join(reports_dir, f"LoadReports-{tribu_id}.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for exp_idx, row in df_exps.iterrows():
            logger1.info(f"\nProcessing: {row.ExpId}")
            df_data, cycles_list = LoadFiles(row.MotorFile, row.DaqFile)
            
            if df_data is None:
                logger1.warning(f"Experiment {row.ExpId} dropped.")
                continue
            
            if np.all(df_data["Current"] == 0):
                logger1.warning(f"Column Current not found in experiment {row.ExpId}. Cannot verify Ohm's Law.")
            else:
                i_theo = df_data["Voltage"] / (row.Req/1e6)
                ratio = df_data["Current"] / i_theo
                tolerance = np.abs(1 - ratio).max()
                logger1.info(f"Experiment {row.ExpId}: Ohm's Law satisfied within {100*tolerance:.0f}% tolerance.")
                
            exp_df = experiment_analysis(row, df_data, cycles_list, pdf)
            
            exp_summary = {
                "ExpId": row.ExpId,
                "TribuId": row.TribuId,
                "Date": row.Date,
                # "Temperature": row.Temperature,
                # "Humidity": row.Humidity,
                "Req": row.Req,
                "NumCycles": len(cycles_list),
                "Duration": df_data.Time.iloc[-1],
                "AvgVoltageMax": exp_df.VoltageMax.mean(),
                "VarVoltageMax": exp_df.VoltageMax.var(),
                "AvgVoltageMin": exp_df.VoltageMin.mean(),
                "VarVoltageMin": exp_df.VoltageMin.var(),
                "AvgPosPeakWidth": exp_df.PosPeakWidth.mean(),
                "VarPosPeakWidth": exp_df.PosPeakWidth.var(),
                "AvgNegPeakWidth": exp_df.NegPeakWidth.mean(),
                "VarNegPeakWidth": exp_df.NegPeakWidth.var(),
                "AvgPosEnergy": exp_df.PosEnergy.mean(),
                "VarPosEnergy": exp_df.PosEnergy.var(),
                "AvgNegEnergy": exp_df.NegEnergy.mean(),
                "VarNegEnergy": exp_df.NegEnergy.var(),
                "AvgTotEnergy": exp_df.TotEnergy.mean(),
                "VarTotEnergy": exp_df.TotEnergy.var()
            }
            exps_summary.append([exp_summary, exp_df])
        
        # Save datasets
        summary_df = pd.DataFrame([d[0] for d in exps_summary])
        excel_path = os.path.join(datasets_dir, f"DataSets-{tribu_id}.xlsx")
        
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            for dicc, df in exps_summary:
                sheet_name = f"Exp_{dicc['ExpId']}"[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for col_idx, col in enumerate(worksheet.columns, 1):
                    max_length = 0
                    column_letter = get_column_letter(col_idx)
                    for cell in col:
                        if cell.value is not None:
                            max_length = max(max_length, len(str(cell.value)))
                    worksheet.column_dimensions[column_letter].width = max_length + 2
        
        # Generate summary plots
        generate_summary_plots(summary_df, pdf)
        
        pdf.close()
        logger1.info(f"/nProcessing completed. PDF saved to {pdf_path} and Excel to {excel_path}.")


if __name__ == "__main__":
    main()










