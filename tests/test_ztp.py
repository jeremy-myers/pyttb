from __future__ import annotations

import numpy as np
import pytest

import pyttb as ttb
from pyttb.gcp.optimizers import LBFGSB, SGD


class TestZTP:
    def test_external_solves(
        self,
    ):
        dense_data = ttb.tenones((2, 2))
        dense_data[0, 1] = 0.0
        dense_data[1, 0] = 0.0
        rank = 2
        optimizer = LBFGSB(maxiter=2)
        result_ztp, initial_guess, info = ttb.ztp(dense_data, rank, optimizer=optimizer)
        assert not result_ztp.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

        # TODO Test with missing data
        # mask = ttb.tenones(dense_data.shape)
        # mask[0] = 0
        # result_ztp, initial_guess, info = ttb.gcp_opt(
        #     dense_data, rank, optimizer, mask=mask
        # )
        # assert not result_ztp.isequal(initial_guess)
        # assert all(initial_guess.weights == 1.0)

    def test_stochastic_solves(
        self,
    ):
        dense_data = ttb.tenones((2, 2))
        dense_data[0, 1] = 0.0
        dense_data[1, 0] = 0.0
        rank = 2

        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        result_ztp, initial_guess, info = ttb.ztp(dense_data, rank, optimizer=optimizer)
        assert not result_ztp.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

        # Providing an initial guess skips the rng to generate initial ktensor
        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        result_ztp, initial_guess, info = ttb.ztp(
            dense_data, rank, optimizer=optimizer, init=initial_guess
        )
        assert not result_ztp.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

        # Test non-normalized initial guess
        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        non_norm_guess = initial_guess.copy()
        non_norm_guess.weights *= 2
        result_ztp, initial_guess, info = ttb.ztp(
            dense_data, rank, optimizer=optimizer, init=non_norm_guess
        )
        assert not result_ztp.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

        # Test just providing factor matrices
        np.random.seed(1)
        optimizer = SGD(max_iters=2, epoch_iters=1)
        result_ztp, initial_guess, info = ttb.ztp(
            dense_data,
            rank,
            optimizer=optimizer,
            init=initial_guess.factor_matrices,
        )
        assert not result_ztp.isequal(initial_guess)
        assert all(initial_guess.weights == 1.0)

    # TODO
    def test_sptensor_with_mask(self):
        # Sptensor with mask
        # with pytest.raises(ValueError):
        #     ttb.ztp(
        #         ttb.sptensor(),
        #         rank,
        #         optimizer,
        #         mask=np.ones((2, 2)),
        #     )
        pass

    def test_invalid_optimizer_options(
        self,
    ):
        dense_data = ttb.tenones((2, 2))
        dense_data[0, 1] = 0.0
        dense_data[1, 0] = 0.0
        rank = 2

        # No mask with stochastic solve
        with pytest.raises(ValueError):
            optimizer = SGD(max_iters=2, epoch_iters=1)
            ttb.ztp(
                dense_data,
                rank,
                optimizer=optimizer,
                mask=ttb.tenones(dense_data.shape),
            )

        # LBFGSB only supports dense
        with pytest.raises(ValueError):
            sparse_data = dense_data.to_sptensor()
            ttb.ztp(sparse_data, rank, LBFGSB())

    def test_general_invalid_options(
        self,
    ):
        dense_data = ttb.tenones((2, 2))
        dense_data[0, 1] = 0.0
        dense_data[1, 0] = 0.0
        rank = 2
        optimizer = SGD(max_iters=2, epoch_iters=1)

        # Incorrect customer objective
        with pytest.raises(ValueError):
            ttb.ztp(dense_data, rank, (1, 2), optimizer)

        # Non-tensor data
        with pytest.raises(ValueError):
            result, initial_guess_custom, info = ttb.ztp([], rank, optimizer=optimizer)

        # Invalid optimizer choices
        with pytest.raises(ValueError):
            ttb.ztp(dense_data, rank, "Not an optimizer")
        # Invalid Init
        with pytest.raises(ValueError):
            optimizer = SGD(max_iters=2, epoch_iters=1)
            ttb.ztp(
                dense_data,
                rank,
                optimizer=optimizer,
                init="Not a supported choice",
            )
