#include <iostream>
#include <chrono>
#include <thread>

// initialisation of OpenMP environment variables
#include <omp.h>

// g++ -fopenmp -O2 6_OpenMP_tasks.cpp -o 6_OpenMP_tasks
// ./6_OpenMP_tasks

// "parallel for" needs a loop whose number of iterations is known when
// the loop starts. Many problems don't look like that : recursive
// algorithms (divide and conquer, tree traversals), linked lists, while
// loops, graphs of jobs depending on each other...
// For them, OpenMP 3.0 introduced TASKS : a task is a piece of work
// (code + its data) that a thread packages and puts in a pool ; any
// thread of the team can then pick it up and execute it, now or later.
// 
// The usual pattern :
//   #pragma omp parallel        -> creates the team of threads
//   #pragma omp single          -> ONE thread creates the tasks...
//   { ... #pragma omp task ...}    ...while the others execute them
//                               -> the implicit barrier at the end of single
//                                  waits until ALL the tasks are done


// ---------------------------------------------------------------
// PART 1 : recursive tasks and granularity -> Fibonacci
// ---------------------------------------------------------------
// fib(n) = fib(n-1) + fib(n-2) is a terribly slow way to compute the
// Fibonacci numbers, but it is the simplest RECURSIVE problem : the
// amount of work is not known in advance, there is no loop to share.

long fib_seq(int n) {
    if (n < 2) return n;
    return fib_seq(n - 1) + fib_seq(n - 2);
}

// each call creates 2 tasks (one per recursive call), then waits for them.
// x and y must be declared shared : in a task, the local variables of the
// enclosing function are FIRSTPRIVATE by default, so the task would write
// its result into its own private copy, which disappears with the task.
long fib_tasks(int n) {
    if (n < 2) return n;

    long x, y;

    #pragma omp task shared(x)
    x = fib_tasks(n - 1);

    #pragma omp task shared(y)
    y = fib_tasks(n - 2);

    // taskwait : wait for the CHILD tasks created above (not for the
    // grandchildren : each child waits for its own children)
    #pragma omp taskwait

    return x + y;
}

// same, with a CUTOFF. Creating, queuing and scheduling a task costs
// around a microsecond, while computing fib(2) costs a nanosecond : below
// a threshold, a task is not worth it and we switch to the sequential
// version. Choosing the granularity of the tasks is THE key to their
// performance. (the clauses if() and final() do a similar job, but less
// efficiently : a task is still created, then executed immediately)
long fib_tasks_cutoff(int n, int cutoff) {
    if (n < 2) return n;
    if (n < cutoff) return fib_seq(n);

    long x, y;

    #pragma omp task shared(x)
    x = fib_tasks_cutoff(n - 1, cutoff);

    #pragma omp task shared(y)
    y = fib_tasks_cutoff(n - 2, cutoff);

    #pragma omp taskwait

    return x + y;
}


// ---------------------------------------------------------------
// PART 2 : traversing a linked list
// ---------------------------------------------------------------
// A linked list can only be walked one node after the other (p = p->next),
// so the number of nodes is unknown and "parallel for" can't share it.
// With tasks : ONE thread walks the list (cheap) and creates one task per
// node ; the processing of the nodes (expensive) is done in parallel.

struct Node {
    int value;
    double result;
    Node* next;
};

// some expensive work on one node (its cost depends on the value)
void process(Node* node) {
    double r = 0.0;
    for (int k = 1; k <= 2000 * node->value; ++k) {
        r += 1.0 / (static_cast<double>(k) * k);
    }
    node->result = r;
}

void process_list_seq(Node* head) {
    for (Node* p = head; p != nullptr; p = p->next) {
        process(p);
    }
}

void process_list_tasks(Node* head) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (Node* p = head; p != nullptr; p = p->next) {
                // p is captured by value (firstprivate, the default) : the
                // task keeps the address of ITS node, even after the loop
                // has moved p to the next one
                #pragma omp task firstprivate(p)
                process(p);
            }
        }   // implicit barrier : every task is finished here
    }
}


// ---------------------------------------------------------------
// PART 3 : task dependencies -> a graph of jobs
// ---------------------------------------------------------------
// depend(in: x)  : this task READS x  -> it waits for the previous task
//                  that writes x
// depend(out: x) : this task WRITES x -> it waits for the previous tasks
//                  that read or write x
// The runtime builds the graph of the tasks from these clauses and runs
// each task as soon as its inputs are ready :
//
//                  A (writes a)
//        .---------'----------.
//        v                    v
//   B (reads a,          C (reads a,        <- B and C run at the same time
//      writes b)            writes c)
//        '---------.----------'
//                  v
//        D (reads b and c, writes d)

// "work" lasting about 'ms' milliseconds
void work(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

void task_graph() {
    int a = 0, b = 0, c = 0, d = 0;
    double t0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp task depend(out: a) shared(a)
        {
            work(200);
            a = 1;
            #pragma omp critical
            std::cout << "  A done at t = " << omp_get_wtime() - t0 << " s" << std::endl;
        }

        #pragma omp task depend(in: a) depend(out: b) shared(a, b)
        {
            work(200);
            b = a + 1;
            #pragma omp critical
            std::cout << "  B done at t = " << omp_get_wtime() - t0 << " s" << std::endl;
        }

        #pragma omp task depend(in: a) depend(out: c) shared(a, c)
        {
            work(200);
            c = a * 10;
            #pragma omp critical
            std::cout << "  C done at t = " << omp_get_wtime() - t0 << " s" << std::endl;
        }

        #pragma omp task depend(in: b, c) depend(out: d) shared(b, c, d)
        {
            work(200);
            d = b + c;
            #pragma omp critical
            std::cout << "  D done at t = " << omp_get_wtime() - t0 << " s" << std::endl;
        }
    }

    std::cout << "  d = " << d << " (expected 12), total = " << omp_get_wtime() - t0
              << " s instead of 0.8 s sequentially" << std::endl;
}


int main() {

    // the first parallel region of a program creates the threads : do it
    // once here so that this cost does not pollute the timings below
    #pragma omp parallel
    { }

    std::cout << "=== Part 1 : recursive tasks (Fibonacci) ===" << std::endl;

    // 1.a : tasks that are too small
    int n = 25;
    double t0 = omp_get_wtime();
    long f_seq = fib_seq(n);
    double t1 = omp_get_wtime();
    std::cout << "fib(" << n << ") sequential           = " << f_seq
              << " | time = " << (t1 - t0) << " s" << std::endl;

    long f_par = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel
    #pragma omp single
    f_par = fib_tasks(n);
    t1 = omp_get_wtime();
    std::cout << "fib(" << n << ") one task per call    = " << f_par
              << " | time = " << (t1 - t0) << " s  <- much SLOWER" << std::endl;

    // 1.b : tasks big enough
    n = 38;
    t0 = omp_get_wtime();
    f_seq = fib_seq(n);
    t1 = omp_get_wtime();
    std::cout << "fib(" << n << ") sequential           = " << f_seq
              << " | time = " << (t1 - t0) << " s" << std::endl;

    t0 = omp_get_wtime();
    #pragma omp parallel
    #pragma omp single
    f_par = fib_tasks_cutoff(n, n - 15);
    t1 = omp_get_wtime();
    std::cout << "fib(" << n << ") tasks with a cutoff  = " << f_par
              << " | time = " << (t1 - t0) << " s" << std::endl;

    std::cout << std::endl << "=== Part 2 : linked list ===" << std::endl;

    // build a list of 2000 nodes with irregular values
    const int nb_nodes = 2000;
    Node* head = nullptr;
    for (int i = 0; i < nb_nodes; ++i) {
        head = new Node{1 + (i * 37) % 50, 0.0, head};
    }

    t0 = omp_get_wtime();
    process_list_seq(head);
    t1 = omp_get_wtime();
    double sum_seq = 0.0;
    for (Node* p = head; p != nullptr; p = p->next) sum_seq += p->result;
    std::cout << "sequential : sum of results = " << sum_seq
              << " | time = " << (t1 - t0) << " s" << std::endl;

    t0 = omp_get_wtime();
    process_list_tasks(head);
    t1 = omp_get_wtime();
    double sum_tasks = 0.0;
    for (Node* p = head; p != nullptr; p = p->next) sum_tasks += p->result;
    std::cout << "tasks      : sum of results = " << sum_tasks
              << " | time = " << (t1 - t0) << " s" << std::endl;

    // free the list (no leak)
    while (head != nullptr) {
        Node* next = head->next;
        delete head;
        head = next;
    }

    std::cout << std::endl << "=== Part 3 : task dependencies ===" << std::endl;
    task_graph();

    return 0;
}
