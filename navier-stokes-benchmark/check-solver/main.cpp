#include <dolfin.h>

using namespace dolfin;

void check_solver()
{
    std::string krylov_method;

    if (!has_linear_algebra_backend("PETSc") && !has_linear_algebra_backend("Tpetra"))
    {
        std::cout << "DOLFIN has not been configured with Trilinos or PETSc. Exiting."
                  << std::endl;
        exit(0);
    }

    if (!has_krylov_solver_preconditioner("amg"))
    {
        std::cout << "Sorry, this demo is only available when DOLFIN is compiled with AMG "
                     "preconditioner, Hypre or ML."
                  << std::endl;
        exit(0);
    }

    if (has_krylov_solver_method("minres"))
    {
        krylov_method = "minres";
    }
    else if (has_krylov_solver_method("tfqmr"))
    {
        krylov_method = "tfqmr";
    }
    else
    {
        std::cout << "Default linear algebra backend was not compiled with MINRES or TFQMR"
                     "Krylov subspace method. Terminating."
                  << std::endl;
        exit(0);
    }

    std::cout << "SOLVER OK!" << std::endl;
}

int main()
{
    check_solver();
}