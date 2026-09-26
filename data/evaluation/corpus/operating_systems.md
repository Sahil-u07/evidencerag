\# Operating Systems



\## Processes



A process is a program in execution. It has its own address space, program counter, registers, stack, heap, and operating-system-managed resources. A process moves through states such as new, ready, running, waiting, and terminated.



\## CPU Scheduling



CPU scheduling determines which ready process receives the processor. Common scheduling strategies include First-Come First-Served, Shortest Job First, Round Robin, and Priority Scheduling.



Round Robin is a preemptive scheduling algorithm designed for time-sharing systems. Each ready process receives a fixed time quantum. When the quantum expires, the running process is preempted and moved to the end of the ready queue.



\## Deadlocks



A deadlock occurs when a collection of processes wait indefinitely for resources held by one another. The four necessary conditions are mutual exclusion, hold and wait, no preemption, and circular wait.



Deadlock avoidance attempts to keep the system in a safe state. The Banker's algorithm evaluates whether granting a resource request can leave the system in a state from which every process can still complete.



\## Virtual Memory



Virtual memory allows a process to use an address space larger than the available physical memory. Paging divides virtual memory into fixed-size pages and physical memory into frames.



A page table maps virtual page numbers to physical frame numbers. When a referenced page is not present in physical memory, a page fault occurs and the operating system must load the page from secondary storage.



\## Page Replacement



When all physical frames are occupied and a new page must be loaded, the operating system selects a victim page for replacement.



FIFO replaces the page that has been in memory the longest. The Optimal algorithm replaces the page whose next use is farthest in the future. LRU replaces the page that has not been used for the longest recent period.



\## Thrashing



Thrashing occurs when a system spends excessive time servicing page faults instead of executing useful work. It can occur when processes demand more memory than the system can effectively provide.



Working-set and page-fault-frequency approaches can be used to help control thrashing.

