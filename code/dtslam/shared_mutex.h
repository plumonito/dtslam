#ifndef SHARED_MUTEX_H
#define SHARED_MUTEX_H

//This has been copied from Boost 1.54 and adapted to use C++11

//  (C) Copyright 2006-8 Anthony Williams
//  (C) Copyright 2012 Vicente J. Botet Escriba
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include "log.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#ifdef _MSC_VER
#define THROW_ERROR(x) std::_Throw_Cpp_error(x)
#else
#define THROW_ERROR(x) std::__throw_system_error(x)
#endif

namespace dtslam
{
class shared_mutex
{
private:
	class state_data
	{
	public:
		state_data() :
				shared_count(0), exclusive(false), upgrade(false), exclusive_waiting_blocked(
						false)
		{
		}

		void assert_free() const
		{
			assert(!exclusive);
			assert(!upgrade);
			assert(shared_count == 0);
		}

		void assert_locked() const
		{
			assert(exclusive);
			assert(shared_count == 0);
			assert(!upgrade);
		}

		void assert_lock_shared() const
		{
			assert(!exclusive);
			assert(shared_count > 0);
			//BOOST_ASSERT( (! upgrade) || (shared_count>1));
			// if upgraded there are at least 2 threads sharing the mutex,
			// except when unlock_upgrade_and_lock has decreased the number of readers but has not taken yet exclusive ownership.
		}

		void assert_lock_upgraded() const
		{
			assert(!exclusive);
			assert(upgrade);
			assert(shared_count > 0);
		}

		void assert_lock_not_upgraded() const
		{
			assert(!upgrade);
		}

		bool can_lock() const
		{
			return !(shared_count || exclusive);
		}

		void exclusive_blocked(bool blocked)
		{
			exclusive_waiting_blocked = blocked;
		}

		void lock()
		{
			exclusive = true;
		}

		void unlock()
		{
			exclusive = false;
			exclusive_waiting_blocked = false;
		}

		bool can_lock_shared() const
		{
			return !(exclusive || exclusive_waiting_blocked);
		}

		bool more_shared() const
		{
			return shared_count > 0;
		}
		unsigned get_shared_count() const
		{
			return shared_count;
		}
		unsigned lock_shared()
		{
			return ++shared_count;
		}

		void unlock_shared()
		{
			--shared_count;
		}

		bool unlock_shared_downgrades()
		{
			if (upgrade)
			{
				upgrade = false;
				exclusive = true;
				return true;
			}
			else
			{
				exclusive_waiting_blocked = false;
				return false;
			}
		}

		void lock_upgrade()
		{
			++shared_count;
			upgrade = true;
		}
		bool can_lock_upgrade() const
		{
			return !(exclusive || exclusive_waiting_blocked || upgrade);
		}

		void unlock_upgrade()
		{
			upgrade = false;
			--shared_count;
		}

		//private:
		unsigned shared_count;
		bool exclusive;
		bool upgrade;
		bool exclusive_waiting_blocked;
	};

	state_data state;
	std::mutex state_change;
	std::condition_variable shared_cond;
	std::condition_variable exclusive_cond;
	std::condition_variable upgrade_cond;

	void release_waiters()
	{
		exclusive_cond.notify_one();
		shared_cond.notify_all();
	}

public:

	//Not copiable
	shared_mutex(shared_mutex const&) = delete;
	shared_mutex& operator=(shared_mutex const&) = delete;

	shared_mutex()
	{
	}

	~shared_mutex()
	{
	}

	void lock_shared()
	{
		std::unique_lock<std::mutex> lk(state_change);
		while (!state.can_lock_shared())
		{
			shared_cond.wait(lk);
		}
		state.lock_shared();
	}

	bool try_lock_shared()
	{
		std::unique_lock<std::mutex> lk(state_change);

		if (!state.can_lock_shared())
		{
			return false;
		}
		state.lock_shared();
		return true;
	}

	template<class Rep, class Period>
	bool try_lock_shared_for(const std::chrono::duration<Rep, Period>& rel_time)
	{
		return try_lock_shared_until(
				std::chrono::steady_clock::now() + rel_time);
	}
	template<class Clock, class Duration>
	bool try_lock_shared_until(
			const std::chrono::time_point<Clock, Duration>& abs_time)
	{
		std::unique_lock<std::mutex> lk(state_change);

		while (!state.can_lock_shared())
		//while(state.exclusive || state.exclusive_waiting_blocked)
		{
			if (std::cv_status::timeout == shared_cond.wait_until(lk, abs_time))
			{
				return false;
			}
		}
		state.lock_shared();
		return true;
	}

	void unlock_shared()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_shared();
		state.unlock_shared();
		if (!state.more_shared())
		{
			if (state.upgrade)
			{
				// As there is a thread doing a unlock_upgrade_and_lock that is waiting for ! state.more_shared()
				// avoid other threads to lock, lock_upgrade or lock_shared, so only this thread is notified.
				state.upgrade = false;
				state.exclusive = true;
				lk.unlock();
				upgrade_cond.notify_one();
			}
			else
			{
				state.exclusive_waiting_blocked = false;
				lk.unlock();
			}
			release_waiters();
		}
	}

	void lock()
	{
		std::unique_lock<std::mutex> lk(state_change);

		while (state.shared_count || state.exclusive)
		{
			state.exclusive_waiting_blocked = true;
			exclusive_cond.wait(lk);
		}
		state.exclusive = true;
	}

	template<class Rep, class Period>
	bool try_lock_for(const std::chrono::duration<Rep, Period>& rel_time)
	{
		return try_lock_until(std::chrono::steady_clock::now() + rel_time);
	}
	template<class Clock, class Duration>
	bool try_lock_until(
			const std::chrono::time_point<Clock, Duration>& abs_time)
	{
		std::unique_lock<std::mutex> lk(state_change);

		while (state.shared_count || state.exclusive)
		{
			state.exclusive_waiting_blocked = true;
			if (std::cv_status::timeout
					== exclusive_cond.wait_until(lk, abs_time))
			{
				if (state.shared_count || state.exclusive)
				{
					state.exclusive_waiting_blocked = false;
					release_waiters();
					return false;
				}
				break;
			}
		}
		state.exclusive = true;
		return true;
	}

	bool try_lock()
	{
		std::unique_lock<std::mutex> lk(state_change);

		if (state.shared_count || state.exclusive)
		{
			return false;
		}
		else
		{
			state.exclusive = true;
			return true;
		}

	}

	void unlock()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_locked();
		state.exclusive = false;
		state.exclusive_waiting_blocked = false;
		state.assert_free();
		release_waiters();
	}

	void lock_upgrade()
	{
		std::unique_lock<std::mutex> lk(state_change);
		while (state.exclusive || state.exclusive_waiting_blocked
				|| state.upgrade)
		{
			shared_cond.wait(lk);
		}
		state.lock_shared();
		state.upgrade = true;
	}

	template<class Rep, class Period>
	bool try_lock_upgrade_for(
			const std::chrono::duration<Rep, Period>& rel_time)
	{
		return try_lock_upgrade_until(
				std::chrono::steady_clock::now() + rel_time);
	}
	template<class Clock, class Duration>
	bool try_lock_upgrade_until(
			const std::chrono::time_point<Clock, Duration>& abs_time)
	{
		std::unique_lock<std::mutex> lk(state_change);
		while (state.exclusive || state.exclusive_waiting_blocked
				|| state.upgrade)
		{
			if (std::cv_status::timeout == shared_cond.wait_until(lk, abs_time))
			{
				if (state.exclusive || state.exclusive_waiting_blocked
						|| state.upgrade)
				{
					return false;
				}
				break;
			}
		}
		state.lock_shared();
		state.upgrade = true;
		return true;
	}

	bool try_lock_upgrade()
	{
		std::unique_lock<std::mutex> lk(state_change);
		if (state.exclusive || state.exclusive_waiting_blocked || state.upgrade)
		{
			return false;
		}
		else
		{
			state.lock_shared();
			state.upgrade = true;
			state.assert_lock_upgraded();
			return true;
		}
	}

	void unlock_upgrade()
	{
		std::unique_lock<std::mutex> lk(state_change);
		//state.upgrade=false;
		state.unlock_upgrade();
		if (!state.more_shared())
		{
			state.exclusive_waiting_blocked = false;
			release_waiters();
		}
		else
		{
			shared_cond.notify_all();
		}
	}

	// Upgrade <-> Exclusive
	void unlock_upgrade_and_lock()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_upgraded();
		state.unlock_shared();
		while (state.more_shared())
		{
			upgrade_cond.wait(lk);
		}
		state.upgrade = false;
		state.exclusive = true;
		state.assert_locked();
	}

	void unlock_and_lock_upgrade()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_locked();
		state.exclusive = false;
		state.upgrade = true;
		state.lock_shared();
		state.exclusive_waiting_blocked = false;
		state.assert_lock_upgraded();
		release_waiters();
	}

	bool try_unlock_upgrade_and_lock()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_upgraded();
		if (!state.exclusive && !state.exclusive_waiting_blocked
				&& state.upgrade && state.shared_count == 1)
		{
			state.shared_count = 0;
			state.exclusive = true;
			state.upgrade = false;
			state.assert_locked();
			return true;
		}
		return false;
	}

	template<class Rep, class Period>
	bool try_unlock_upgrade_and_lock_for(
			const std::chrono::duration<Rep, Period>& rel_time)
	{
		return try_unlock_upgrade_and_lock_until(
				std::chrono::steady_clock::now() + rel_time);
	}
	template<class Clock, class Duration>
	bool try_unlock_upgrade_and_lock_until(
			const std::chrono::time_point<Clock, Duration>& abs_time)
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_upgraded();
		if (state.shared_count != 1)
		{
			for (;;)
			{
				std::cv_status status = shared_cond.wait_until(lk, abs_time);
				if (state.shared_count == 1)
					break;
				if (status == std::cv_status::timeout)
					return false;
			}
		}
		state.upgrade = false;
		state.exclusive = true;
		state.exclusive_waiting_blocked = false;
		state.shared_count = 0;
		return true;
	}

	// Shared <-> Exclusive
	void unlock_and_lock_shared()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_locked();
		state.exclusive = false;
		state.lock_shared();
		state.exclusive_waiting_blocked = false;
		release_waiters();
	}

	bool try_unlock_shared_and_lock()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_shared();
		if (!state.exclusive && !state.exclusive_waiting_blocked
				&& !state.upgrade && state.shared_count == 1)
		{
			state.shared_count = 0;
			state.exclusive = true;
			return true;
		}
		return false;
	}

	template<class Rep, class Period>
	bool try_unlock_shared_and_lock_for(
			const std::chrono::duration<Rep, Period>& rel_time)
	{
		return try_unlock_shared_and_lock_until(
				std::chrono::steady_clock::now() + rel_time);
	}
	template<class Clock, class Duration>
	bool try_unlock_shared_and_lock_until(
			const std::chrono::time_point<Clock, Duration>& abs_time)
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_shared();
		if (state.shared_count != 1)
		{
			for (;;)
			{
				std::cv_status status = shared_cond.wait_until(lk, abs_time);
				if (state.shared_count == 1)
					break;
				if (status == std::cv_status::timeout)
					return false;
			}
		}
		state.upgrade = false;
		state.exclusive = true;
		state.exclusive_waiting_blocked = false;
		state.shared_count = 0;
		return true;
	}

	// Shared <-> Upgrade
	void unlock_upgrade_and_lock_shared()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_upgraded();
		state.upgrade = false;
		state.exclusive_waiting_blocked = false;
		release_waiters();
	}

	bool try_unlock_shared_and_lock_upgrade()
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_shared();
		if (!state.exclusive && !state.exclusive_waiting_blocked
				&& !state.upgrade)
		{
			state.upgrade = true;
			return true;
		}
		return false;
	}

	template<class Rep, class Period>
	bool try_unlock_shared_and_lock_upgrade_for(
			const std::chrono::duration<Rep, Period>& rel_time)
	{
		return try_unlock_shared_and_lock_upgrade_until(
				std::chrono::steady_clock::now() + rel_time);
	}
	template<class Clock, class Duration>
	bool try_unlock_shared_and_lock_upgrade_until(
			const std::chrono::time_point<Clock, Duration>& abs_time)
	{
		std::unique_lock<std::mutex> lk(state_change);
		state.assert_lock_shared();
		if (state.exclusive || state.exclusive_waiting_blocked || state.upgrade)
		{
			for (;;)
			{
				std::cv_status status = exclusive_cond.wait_until(lk, abs_time);
				if (!state.exclusive && !state.exclusive_waiting_blocked
						&& !state.upgrade)
					break;
				if (status == std::cv_status::timeout)
					return false;
			}
		}
		state.upgrade = true;
		return true;
	}
};

typedef shared_mutex upgrade_mutex;

///////////////////////////////////////////////////////////////////////////////////////////////
// This copied and modified from the C++11 std library

/// @brief  Scoped lock idiom.
// Acquire the mutex here with a constructor call, then release with
// the destructor call in accordance with RAII style.
template<typename _Mutex>
class shared_lock_guard
{
public:
	typedef _Mutex mutex_type;

	explicit shared_lock_guard(mutex_type& __m) :
			_M_device(__m)
	{
		_M_device.lock_shared();
	}

	shared_lock_guard(mutex_type& __m, std::adopt_lock_t) :
			_M_device(__m)
	{
	} // calling thread owns mutex

	~shared_lock_guard()
	{
		_M_device.unlock_shared();
	}

	shared_lock_guard(const shared_lock_guard&) = delete;
	shared_lock_guard& operator=(const shared_lock_guard&) = delete;

private:
	mutex_type& _M_device;
};

/// @brief  Scoped lock idiom.
// Acquire the mutex here with a constructor call, then release with
// the destructor call in accordance with RAII style.
template<typename _Mutex>
class upgrade_lock_guard
{
public:
	typedef _Mutex mutex_type;

	explicit upgrade_lock_guard(mutex_type& __m) :
			_M_device(__m)
	{
		_M_device.lock_upgrade();
	}

	upgrade_lock_guard(mutex_type& __m, std::adopt_lock_t) :
			_M_device(__m)
	{
	} // calling thread owns mutex

	~upgrade_lock_guard()
	{
		_M_device.unlock_upgrade();
	}

	upgrade_lock_guard(const upgrade_lock_guard&) = delete;
	upgrade_lock_guard& operator=(const upgrade_lock_guard&) = delete;

private:
	mutex_type& _M_device;
};

/// shared_lock
template<typename _Mutex>
class shared_lock
{
public:
	typedef _Mutex mutex_type;

	shared_lock()
	: _M_device(0), _M_owns(false)
	{}

	explicit shared_lock(mutex_type& __m)
	: _M_device(&__m), _M_owns(false)
	{
		lock();
		_M_owns = true;
	}

	shared_lock(mutex_type& __m, std::defer_lock_t) 
	: _M_device(&__m), _M_owns(false)
	{}

	shared_lock(mutex_type& __m, std::try_to_lock_t)
	: _M_device(&__m), _M_owns(_M_device->try_lock_shared())
	{}

	shared_lock(mutex_type& __m, std::adopt_lock_t)
	: _M_device(&__m), _M_owns(true)
	{
		// XXX calling thread owns mutex
	}

	template<typename _Clock, typename _Duration>
	shared_lock(mutex_type& __m,
			const std::chrono::time_point<_Clock, _Duration>& __atime)
	: _M_device(&__m), _M_owns(_M_device->try_lock_shared_until(__atime))
	{}

	template<typename _Rep, typename _Period>
	shared_lock(mutex_type& __m,
			const std::chrono::duration<_Rep, _Period>& __rtime)
	: _M_device(&__m), _M_owns(_M_device->try_lock_shared_for(__rtime))
	{}

	~shared_lock()
	{
		if (_M_owns)
		unlock();
	}

	shared_lock(const shared_lock&) = delete;
	shared_lock& operator=(const shared_lock&) = delete;

	shared_lock(shared_lock&& __u) 
	: _M_device(__u._M_device), _M_owns(__u._M_owns)
	{
		__u._M_device = 0;
		__u._M_owns = false;
	}

	shared_lock& operator=(shared_lock&& __u) 
	{
		if(_M_owns)
		unlock();

		shared_lock(std::move(__u)).swap(*this);

		__u._M_device = 0;
		__u._M_owns = false;

		return *this;
	}

	void
	lock()
	{
		if (!_M_device)
			THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_owns)
			THROW_ERROR(int(std::errc::resource_deadlock_would_occur));
		else
		{
			_M_device->lock_shared();
			_M_owns = true;
		}
	}

	bool
	try_lock()
	{
		if (!_M_device)
			THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_owns)
			THROW_ERROR(int(std::errc::resource_deadlock_would_occur));
		else
		{
			_M_owns = _M_device->try_lock_shared();
			return _M_owns;
		}
	}

	template<typename _Clock, typename _Duration>
	bool
	try_lock_until(const std::chrono::time_point<_Clock, _Duration>& __atime)
	{
		if (!_M_device)
		THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_owns)
		THROW_ERROR(int(std::errc::resource_deadlock_would_occur));
		else
		{
			_M_owns = _M_device->try_lock_shared_until(__atime);
			return _M_owns;
		}
	}

	template<typename _Rep, typename _Period>
	bool
	try_lock_for(const std::chrono::duration<_Rep, _Period>& __rtime)
	{
		if (!_M_device)
		THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_owns)
		THROW_ERROR(int(std::errc::resource_deadlock_would_occur));
		else
		{
			_M_owns = _M_device->try_lock_shared_for(__rtime);
			return _M_owns;
		}
	}

	void
	unlock()
	{
		if (!_M_owns)
		THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_device)
		{
			_M_device->unlock_shared();
			_M_owns = false;
		}
	}

	void
	swap(shared_lock& __u) 
	{
		std::swap(_M_device, __u._M_device);
		std::swap(_M_owns, __u._M_owns);
	}

	mutex_type*
	release() 
	{
		mutex_type* __ret = _M_device;
		_M_device = 0;
		_M_owns = false;
		return __ret;
	}

	bool
	owns_lock() const 
	{	return _M_owns;}

	explicit operator bool() const 
	{	return owns_lock();}

	mutex_type*
	mutex() const 
	{	return _M_device;}

private:
	mutex_type* _M_device;
	bool _M_owns; // XXX use atomic_bool
};

/// upgrade_lock
template<typename _Mutex>
class upgrade_lock
{
public:
	typedef _Mutex mutex_type;

	upgrade_lock() 
	: _M_device(0), _M_owns(false)
	{}

	explicit upgrade_lock(mutex_type& __m)
	: _M_device(&__m), _M_owns(false)
	{
		lock();
		_M_owns = true;
	}

	upgrade_lock(mutex_type& __m, std::defer_lock_t) 
	: _M_device(&__m), _M_owns(false)
	{}

	upgrade_lock(mutex_type& __m, std::try_to_lock_t)
	: _M_device(&__m), _M_owns(_M_device->try_lock_upgrade())
	{}

	upgrade_lock(mutex_type& __m, std::adopt_lock_t)
	: _M_device(&__m), _M_owns(true)
	{
		// XXX calling thread owns mutex
	}

	template<typename _Clock, typename _Duration>
	upgrade_lock(mutex_type& __m,
			const std::chrono::time_point<_Clock, _Duration>& __atime)
	: _M_device(&__m), _M_owns(_M_device->try_lock_upgrade_until(__atime))
	{}

	template<typename _Rep, typename _Period>
	upgrade_lock(mutex_type& __m,
			const std::chrono::duration<_Rep, _Period>& __rtime)
	: _M_device(&__m), _M_owns(_M_device->try_lock_upgrade_for(__rtime))
	{}

	~upgrade_lock()
	{
		if (_M_owns)
		unlock();
	}

	upgrade_lock(const upgrade_lock&) = delete;
	upgrade_lock& operator=(const upgrade_lock&) = delete;

	upgrade_lock(upgrade_lock&& __u) 
	: _M_device(__u._M_device), _M_owns(__u._M_owns)
	{
		__u._M_device = 0;
		__u._M_owns = false;
	}

	upgrade_lock& operator=(upgrade_lock&& __u) 
	{
		if(_M_owns)
		unlock();

		upgrade_lock(std::move(__u)).swap(*this);

		__u._M_device = 0;
		__u._M_owns = false;

		return *this;
	}

	void
	lock()
	{
		if (!_M_device)
		THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_owns)
		THROW_ERROR(int(std::errc::resource_deadlock_would_occur));
		else
		{
			_M_device->lock_upgrade();
			_M_owns = true;
		}
	}

	bool
	try_lock()
	{
		if (!_M_device)
		THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_owns)
		THROW_ERROR(int(std::errc::resource_deadlock_would_occur));
		else
		{
			_M_owns = _M_device->try_lock_upgrade();
			return _M_owns;
		}
	}

	template<typename _Clock, typename _Duration>
	bool
	try_lock_until(const std::chrono::time_point<_Clock, _Duration>& __atime)
	{
		if (!_M_device)
		THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_owns)
		THROW_ERROR(int(std::errc::resource_deadlock_would_occur));
		else
		{
			_M_owns = _M_device->try_lock_upgrade_until(__atime);
			return _M_owns;
		}
	}

	template<typename _Rep, typename _Period>
	bool
	try_lock_for(const std::chrono::duration<_Rep, _Period>& __rtime)
	{
		if (!_M_device)
		THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_owns)
		THROW_ERROR(int(std::errc::resource_deadlock_would_occur));
		else
		{
			_M_owns = _M_device->try_lock_upgrade_for(__rtime);
			return _M_owns;
		}
	}

	void
	unlock()
	{
		if (!_M_owns)
		THROW_ERROR(int(std::errc::operation_not_permitted));
		else if (_M_device)
		{
			_M_device->unlock_upgrade();
			_M_owns = false;
		}
	}

	void
	swap(upgrade_lock& __u) 
	{
		std::swap(_M_device, __u._M_device);
		std::swap(_M_owns, __u._M_owns);
	}

	mutex_type*
	release() 
	{
		mutex_type* __ret = _M_device;
		_M_device = 0;
		_M_owns = false;
		return __ret;
	}

	bool
	owns_lock() const 
	{	return _M_owns;}

	explicit operator bool() const 
	{	return owns_lock();}

	mutex_type*
	mutex() const 
	{	return _M_device;}

private:
	mutex_type* _M_device;
	bool _M_owns; // XXX use atomic_bool
};
}

#endif
