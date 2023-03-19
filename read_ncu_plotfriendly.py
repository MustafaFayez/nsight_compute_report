import csv
import pdb
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser("script to read ncu output csv file and create file containing memory bound nature for m=n gemms")
parser.add_argument('--m', action='store', default=[4],type=str)
parser.add_argument('--n', action='store', default=-1,type=int)
parser.add_argument('--k', action='store', default=[4],type=str)
parser.add_argument('--l', action='store', default='dram', choices=['dram', 'l2', 'l1', 'mbu1', 'mbu2', 'mbu3', 'cru'])
parser.add_argument('--t', action='store', default='old', choices=['new', 'old', 'sol'])
parser.add_argument('--direc', action='store', default='./rtx2080_roofline_results_fp16_tanvi/1/')
parser.add_argument('--out', action='store', default='out_dram.csv')
parser.add_argument('--auto', action='store', default=False,type=bool)

args = parser.parse_args()

memory_level = args.l
direc = args.direc
m_list=args.m.split(',')
n_list=args.n
k_list=args.k.split(',')
type=args.t
naming=args.auto

if naming == True:
    out_file = direc+"/plot_membound_"+str(type)+"_"+str(memory_level)+"_n"+str(n_list)+".csv"
else:
    out_file = args.out

fout = open(out_file, "w")
fout.write(out_file+"\n")
fout.write("k\m=n,")
for m in m_list:
    fout.write(str(m)+", ")
fout.write("\n")

#import pdb; pdb.set_trace() 
for k in k_list:
    fout.write(str(k)+", ")
    for m in m_list:
        # fout.write(str(m))
        status='NaN'
        if n_list == -1:
            csv_name = "log_"+str(m)+"_"+str(m)+"_"+str(k)+".csv"
        else:
            csv_name = "log_"+str(m)+"_"+str(n_list)+"_"+str(k)+".csv"
        csv_file_name = direc + "/" + csv_name
        with open(csv_file_name, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row)>26:
                    if row[0].find('ID') > -1:
                        if type == 'sol':
                            #Metrics from Speed of Light Analysis in Nsight Compute
                            CompThru = row.index('sm__throughput.avg.pct_of_peak_sustained_elapsed')
                            DRAMThru = row.index('gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed')
                            L2Thru = row.index('lts__throughput.avg.pct_of_peak_sustained_elapsed')
                            L1Thru = row.index('l1tex__throughput.avg.pct_of_peak_sustained_active')
                        else:
                            TensorUtil_index = row.index('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed')
                            FMAUtil_index = row.index('sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active')
                            ALUUtil_index = row.index('sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active')
                            FP64Util_index = row.index('sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active')
                            FP16Util_index = row.index('sm__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active')
                            XUUtil_index = row.index('sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active')
                            DRAMUtil_index = row.index('gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed')
                            DRAMThroughput_index = row.index('dram__bytes.sum.per_second')
                            L2HitRate_index = row.index('lts__t_sector_hit_rate.pct')
                            L1HitRate_index = row.index('l1tex__t_sector_hit_rate.pct')

                            PeakWorkPerCycle_index = row.index('sm__inst_executed_pipe_tensor.sum.peak_sustained')
                            CyclePerSecond_index = row.index('sm__cycles_elapsed.avg.per_second')
                            PeakDRAMPerCycle_index = row.index('dram__bytes.sum.peak_sustained')
                            DRAMCyclesPerSecond_index = row.index('dram__cycles_elapsed.avg.per_second')
                            AchievedWorkPerCycles_index= row.index('smsp__inst_executed_pipe_tensor.sum.per_cycle_elapsed')
                            AchievedCyclesPerSecond_index = row.index('smsp__cycles_elapsed.avg.per_second')
                            AchievedDRAMTraffic_index = row.index('dram__bytes.sum.per_second')

                            PeakL2PerCycle_index = row.index('lts__t_bytes.sum.peak_sustained')
                            L2CyclesPerSecond_index = row.index('lts__cycles_elapsed.avg.per_second')
                            PeakL1PerCycle_index = row.index('l1tex__t_bytes.sum.peak_sustained')
                            L1CyclesPerSecond_index = row.index('l1tex__cycles_elapsed.avg.per_second')
                            AchievedL2Traffic_index = row.index('lts__t_bytes.sum.per_second')
                            AchievedL1Traffic_index = row.index('l1tex__t_bytes.sum.per_second')



                    if row[4].find('gemm') > -1:
                        status='true'
                # based on GPU model and operation datatype we decide the performance factor based on this issue in nsight compute:
                # https://forums.developer.nvidia.com/t/why-the-compute-throughputs-value-is-different-from-the-actual-performance-peak-performance/227563
                # from the datasheet I found these factors.
                # also the L1 BW calc was wrong as per this issue https://forums.developer.nvidia.com/t/confused-about-the-l1-smem-bw-reported-by-nsight-compute-hierarchical-roofline-plots/232524/3
                # so i took the fixed numbers from datasheets
                        if row[4].find('turing') > -1:
                            #print("turing")
                            sf = 512
                            peakl1_BW =  14*1024*1024*1024*1024
                        elif row[4].find('ampere') > -1 or row[4].find('sm80') > -1:
                            peakl1_BW =  19*1024*1024*1024*1024
                            if row[4].find('int8') > -1 or row[4].find('imma') > -1:
                                sf = 4096
                            else:
                                sf = 2048
                        else: #default is turing
                            peakl1_BW = 14*1024*1024*1024*1024
                            sf = 512
                        if type == 'sol':
                            compthru = float(row[CompThru])
                            dramthru = float(row[DRAMThru])
                            l2thru = float(row[L2Thru])
                            l1thru = float(row[L1Thru])

                            match memory_level:
                                case 'dram':
                                    fout.write(str(compthru < dramthru)+ ', ')
                                    break
                                case 'l2':
                                    fout.write(str(compthru < l2thru)+ ', ')
                                    break
                                case 'l1':
                                    fout.write(str(compthru < l1thru)+ ', ')
                                    break
                                case 'cru': #compute resources utilization
                                    fout.write(str(compthru)+ ', ')
                                    break
                                case 'mbu1':
                                    fout.write(str(dramthru)+ ', ')
                                    break
                                case 'mbu2':
                                    fout.write(str(l2thru)+ ', ')
                                    break
                                case 'mbu3':
                                    fout.write(str(l1thru)+ ', ')
                                    break
                        elif type == 'new':
                            mbu1 = float(row[DRAMThroughput_index].replace(",",""))/(616*1e7)
                            mbu2 = float(row[DRAMUtil_index])
                            tcu  = 0.01*float(row[TensorUtil_index])
                            fmau  = 0.01*float(row[FMAUtil_index])
                            aluu  = 0.01*float(row[ALUUtil_index])
                            match memory_level:
                                case 'dram':
                                    fout.write(str(mbu1 > tcu)+ ", ")
                                    break
                                case 'cru':
                                    fout.write(str(tcu)+ ", ")
                                    break
                                case 'mbu1':
                                    fout.write(str(mbu1)+ ", ")
                                    break
                                case 'mbu2':
                                    fout.write(str(mbu2)+ ", ")
                                    break
                        else:
                            peakperfpercycle = row[PeakWorkPerCycle_index]
                            peakworkcyclespersecond = row[CyclePerSecond_index]
                            peak_perf = float(peakperfpercycle.replace(",","")) * float(peakworkcyclespersecond.replace(",",""))*sf
                            peakdrambytes = row[PeakDRAMPerCycle_index]
                            peakdramcyclespersecond = row[DRAMCyclesPerSecond_index]

                            peak_BW = float(peakdrambytes.replace(",","")) * float(peakdramcyclespersecond.replace(",",""))
                            achievedworkpercycle = row[AchievedWorkPerCycles_index]
                            achievedcyclespersecond = row[AchievedCyclesPerSecond_index]
                            achievedtraffic = row[AchievedDRAMTraffic_index]
                            kernel_perf = float(achievedworkpercycle.replace(",","")) * float(achievedcyclespersecond.replace(",",""))*sf
                            ridge_point = peak_perf / peak_BW
                            arithmetic_intensity = kernel_perf / float(achievedtraffic.replace(",",""))

                            peakl2bytes = row[PeakL2PerCycle_index]
                            peakl2cyclespersecond = row[L2CyclesPerSecond_index]
                            peakl2_BW =  float(peakl2bytes.replace(",","")) * float(peakl2cyclespersecond.replace(",",""))
                            achievedl2traffic = row[AchievedL2Traffic_index]
                            ridge_point_l2 = peak_perf / peakl2_BW
                            arithmetic_intensity_l2 = kernel_perf / float(achievedl2traffic.replace(",",""))
                            peakl1bytes = row[PeakL1PerCycle_index]	
                            peakl1cyclespersecond = row[L1CyclesPerSecond_index]
                            #peakl1_BW =  float(peakl1bytes.replace(",","")) * float(peakl1cyclespersecond.replace(",",""))
                            achievedl1traffic = row[AchievedL1Traffic_index]
                            ridge_point_l1 = peak_perf / peakl1_BW
                            arithmetic_intensity_l1 = kernel_perf / float(achievedl1traffic.replace(",",""))

                        
                            # import pdb; pdb.set_trace()
                            if  memory_level == 'dram':
                                fout.write(str(arithmetic_intensity < ridge_point) + ", " )
                            elif memory_level == 'l2':
                                fout.write(str(arithmetic_intensity_l2 < ridge_point_l2) + ", " )
                            else: #l1 level
                                fout.write(str(arithmetic_intensity_l1 < ridge_point_l1) + ", " )

        if status != 'true':
            fout.write(status+', ') 
    fout.write("\n")
fout.close()
